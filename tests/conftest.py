"""Make the repository root importable and provide a headless app fixture."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake"


class FakeResp:
    def __init__(self):
        self.files = [("image_0.png", FAKE_PNG)]


def wait_for_prefetch(a, timeout=10.0):
    t = a.prefetcher._thread
    if t is not None:
        t.join(timeout)


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Redirect every data path to tmp, mock the network, and return a builder.

    build(history=None) constructs a real ArtistELORanker. Pass a list of
    history records to pre-seed comparison_history.json before construction.
    """
    import artist_elo_ranker as ranker

    tags = tmp_path / "tags.txt"
    tags.write_text("\n".join(f"artist{i}" for i in range(40)) + "\n")
    monkeypatch.setattr(ranker, "ARTIST_TAGS_FILE", tags)
    monkeypatch.setattr(ranker, "ELO_RATINGS_FILE", tmp_path / "ratings.json")
    monkeypatch.setattr(ranker, "COMPARISON_HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(ranker, "ACTIVE_POOL_FILE", tmp_path / "pool.json")
    monkeypatch.setattr(ranker, "COMPARISON_IMAGES_DIR", tmp_path / "images")
    if hasattr(ranker, "SKILL_FILE"):
        monkeypatch.setattr(ranker, "SKILL_FILE", tmp_path / "skill.json")
    monkeypatch.setenv("NOVELAI_API_KEY", "pst-test-key")

    payloads = []

    async def fake_request(self, session=None, **kwargs):
        payloads.append(self.model_dump(mode="json", exclude_none=True))
        return FakeResp()

    monkeypatch.setattr(ranker.GenerateImageInfer, "request", fake_request)
    built = []

    def build(history=None):
        if history is not None:
            (tmp_path / "history.json").write_text(json.dumps(history))
        a = ranker.ArtistELORanker()
        a._payloads = payloads
        a._tmp = tmp_path
        built.append(a)
        return a

    yield build
    # Let in-flight prefetches finish while the network mock is still in place.
    for a in built:
        wait_for_prefetch(a)


@pytest.fixture
def app(app_env):
    return app_env()
