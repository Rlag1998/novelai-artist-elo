"""
Headless end-to-end flow through the real ArtistELORanker: first round generated
on demand, second round served from the prefetch, a pick recorded with seeds and
prompts, undo restoring the pair, and the Gradio UI building without a server.

All data paths are redirected to a temp directory. The network call is mocked.
"""
import json
import time

import pytest

import artist_elo_ranker as ranker

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake"


class _FakeResp:
    def __init__(self):
        self.files = [("image_0.png", FAKE_PNG)]


@pytest.fixture
def app(tmp_path, monkeypatch):
    tags = tmp_path / "tags.txt"
    tags.write_text("\n".join(f"artist{i}" for i in range(40)) + "\n")
    monkeypatch.setattr(ranker, "ARTIST_TAGS_FILE", tags)
    monkeypatch.setattr(ranker, "ELO_RATINGS_FILE", tmp_path / "ratings.json")
    monkeypatch.setattr(ranker, "COMPARISON_HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(ranker, "ACTIVE_POOL_FILE", tmp_path / "pool.json")
    monkeypatch.setattr(ranker, "COMPARISON_IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(ranker, "ACTIVE_POOL_SIZE", 20)
    monkeypatch.setenv("NOVELAI_API_KEY", "pst-test-key")

    payloads = []

    async def fake_request(self, session=None, **kwargs):
        payloads.append(self.model_dump(mode="json", exclude_none=True))
        return _FakeResp()

    monkeypatch.setattr(ranker.GenerateImageInfer, "request", fake_request)
    a = ranker.ArtistELORanker()
    a._payloads = payloads
    yield a
    # Let any in-flight prefetch finish while the network mock is still in place,
    # so no thread outlives the test and reaches the real API.
    _wait_for_prefetch(a)


def _wait_for_prefetch(a, timeout=10.0):
    t = a.prefetcher._thread
    if t is not None:
        t.join(timeout)


def test_first_round_generates_and_second_is_served_from_prefetch(app):
    out = app.generate_new_comparison("", "", True, 0)
    assert out[0] and out[1] and "generated" in out[2].lower()
    assert len(app._payloads) == 2
    seeds = {p["parameters"]["seed"] for p in app._payloads}
    assert len(seeds) == 1, "shared seed within the round"
    assert app.current_pair is not None and app.current_pair.seed_a in seeds

    _wait_for_prefetch(app)
    assert len(app._payloads) == 4, "the next pair was prefetched in the background"

    out2 = app.generate_new_comparison("", "", True, 0)
    assert "ready" in out2[2].lower()
    assert len(app._payloads) >= 4
    assert out2[0] != out[0], "a different pair was shown"


def test_changing_settings_discards_the_prefetched_pair(app):
    app.generate_new_comparison("", "", True, 0)
    _wait_for_prefetch(app)
    n = len(app._payloads)
    out = app.generate_new_comparison("1boy, castle, {artist_placeholder}", "lowres", False, 2)
    assert "generated" in out[2].lower(), "settings changed, so a fresh pair was made on demand"
    assert len(app._payloads) >= n + 2
    assert app.current_pair.uc_preset == 2 and app.current_pair.quality_toggle is False
    assert "1boy, castle" in app.current_pair.prompt_a


def test_pick_records_full_round_and_undo_restores_it(app, tmp_path):
    app.generate_new_comparison("", "", True, 0)
    pair = app.current_pair
    app.pick_winner("A")

    records = json.loads((tmp_path / "history.json").read_text())
    assert len(records) == 1
    rec = records[0]
    assert rec["winner"] == "A"
    assert rec["seed_a"] == pair.seed_a and rec["seed_b"] == pair.seed_b
    assert rec["prompt_a"] == pair.prompt_a and rec["prompt_b"] == pair.prompt_b
    assert rec["model_id"] == ranker.MODEL_ID and rec["uc_preset"] == 0 and rec["quality_toggle"] is True
    assert "Side bias" in app.format_top_artists_display()

    ratings = json.loads((tmp_path / "ratings.json").read_text())
    assert ratings["comparison_count"] == 1

    app.generate_new_comparison("", "", True, 0)  # move on, then undo the pick
    app.undo_last_selection()
    assert app.current_pair is pair
    assert json.loads((tmp_path / "history.json").read_text()) == []
    assert json.loads((tmp_path / "ratings.json").read_text())["comparison_count"] == 0


def test_ui_builds_headless(app):
    blocks = app.create_ui()
    assert blocks is not None
