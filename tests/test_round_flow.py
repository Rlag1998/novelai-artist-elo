"""
Fail-first tests for Batch A: shared seed per pair, background prefetch of the
next pair, informative NovelAI errors with retry, and a comparison history that
records everything needed to reproduce a round plus a side-bias counter.

Nothing here touches the network or the real data files.
"""
import asyncio
import json
import logging
import threading
import time

import pytest

import config
import artist_elo_ranker as ranker

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake"


class _FakeResp:
    def __init__(self, data=FAKE_PNG):
        self.files = [("image_0.png", data)]


class _Boom(Exception):
    """Mimics novelai_python errors: str(e) is empty, details live on attributes."""

    def __init__(self, code=None, message=""):
        super().__init__()
        self.code = code
        self.message = message


def _capture_requests(monkeypatch, fail_plan=None):
    """Patch the network call. Raises the planned exceptions first, then succeeds."""
    calls = {"attempts": 0, "payloads": []}
    plan = list(fail_plan or [])

    async def fake_request(self, session=None, **kwargs):
        calls["attempts"] += 1
        if plan:
            raise plan.pop(0)
        calls["payloads"].append(self.model_dump(mode="json", exclude_none=True))
        return _FakeResp()

    monkeypatch.setattr(ranker.GenerateImageInfer, "request", fake_request)
    return calls


@pytest.fixture
def tags_manager(tmp_path):
    f = tmp_path / "tags.txt"
    f.write_text("alpha\nbravo\ncharlie\ndelta\necho\n")
    return ranker.ArtistTagManager(f)  # no active pool: pure random, no file writes


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(ranker, "_retry_sleep", fake_sleep)
    return slept


# ── A1: shared seed per pair ─────────────────────────────────────────

def test_seed_mode_defaults_to_shared():
    assert config.SEED_MODE == "shared"


def test_shared_seed_mode_gives_both_images_the_same_seed(tmp_path, monkeypatch, tags_manager):
    calls = _capture_requests(monkeypatch)
    pair = asyncio.run(ranker.generate_comparison_pair(
        "1girl, {artist_placeholder}", tags_manager, None, tmp_path, "lowres", True, 0, seed_mode="shared"))
    assert pair is not None and pair.path_a.exists() and pair.path_b.exists()
    seeds = [p["parameters"]["seed"] for p in calls["payloads"]]
    assert len(seeds) == 2 and seeds[0] == seeds[1]
    assert pair.seed_a == pair.seed_b == seeds[0]


def test_independent_seed_mode_gives_different_seeds(tmp_path, monkeypatch, tags_manager):
    calls = _capture_requests(monkeypatch)
    pair = asyncio.run(ranker.generate_comparison_pair(
        "1girl, {artist_placeholder}", tags_manager, None, tmp_path, "lowres", True, 0, seed_mode="independent"))
    seeds = [p["parameters"]["seed"] for p in calls["payloads"]]
    assert len(seeds) == 2 and seeds[0] != seeds[1]
    assert pair.seed_a == seeds[0] and pair.seed_b == seeds[1]


def test_pair_records_everything_needed_to_reproduce(tmp_path, monkeypatch, tags_manager):
    calls = _capture_requests(monkeypatch)
    pair = asyncio.run(ranker.generate_comparison_pair(
        "1girl, {artist_placeholder}", tags_manager, None, tmp_path, "lowres", False, 2, seed_mode="shared"))
    assert calls["payloads"][0]["input"].startswith(pair.prompt_a)
    assert calls["payloads"][1]["input"].startswith(pair.prompt_b)
    assert all(f"artist: {a}" in pair.prompt_a for a in pair.artists_a)
    assert set(pair.artists_a) != set(pair.artists_b)
    assert pair.base_prompt == "1girl, {artist_placeholder}"
    assert pair.negative_prompt == "lowres" and pair.quality_toggle is False and pair.uc_preset == 2
    assert pair.model_id == config.MODEL_ID == calls["payloads"][0]["model"]


def test_failed_generation_returns_none(tmp_path, monkeypatch, tags_manager, no_sleep):
    _capture_requests(monkeypatch, fail_plan=[_Boom(400, "bad request")])
    pair = asyncio.run(ranker.generate_comparison_pair(
        "1girl, {artist_placeholder}", tags_manager, None, tmp_path))
    assert pair is None


# ── A3: informative errors and retry ─────────────────────────────────

def test_429_is_retried_and_logged_with_detail(tmp_path, monkeypatch, caplog, no_sleep):
    calls = _capture_requests(monkeypatch, fail_plan=[_Boom(429, "concurrent generation"), _Boom(429, "concurrent generation")])
    with caplog.at_level(logging.WARNING, logger="artist_elo_ranker"):
        ok = asyncio.run(ranker.generate_image(None, "1girl", tmp_path / "x.png"))
    assert ok is True
    assert calls["attempts"] == 3 and len(calls["payloads"]) == 1
    assert len(no_sleep) == 2
    assert "_Boom" in caplog.text and "429" in caplog.text and "concurrent generation" in caplog.text


def test_client_error_is_not_retried_but_fully_logged(tmp_path, monkeypatch, caplog, no_sleep):
    calls = _capture_requests(monkeypatch, fail_plan=[_Boom(400, "validation failed")])
    with caplog.at_level(logging.WARNING, logger="artist_elo_ranker"):
        ok = asyncio.run(ranker.generate_image(None, "1girl", tmp_path / "x.png"))
    assert ok is False and calls["attempts"] == 1 and no_sleep == []
    assert "_Boom" in caplog.text and "400" in caplog.text and "validation failed" in caplog.text


def test_blank_exception_is_still_informative(tmp_path, monkeypatch, caplog, no_sleep):
    _capture_requests(monkeypatch, fail_plan=[_Boom(None, "")])
    with caplog.at_level(logging.WARNING, logger="artist_elo_ranker"):
        ok = asyncio.run(ranker.generate_image(None, "1girl", tmp_path / "x.png"))
    assert ok is False
    assert "_Boom" in caplog.text and "(no message)" in caplog.text


def test_retry_gives_up_after_max_attempts(tmp_path, monkeypatch, no_sleep):
    calls = _capture_requests(monkeypatch, fail_plan=[_Boom(429, "busy")] * 10)
    ok = asyncio.run(ranker.generate_image(None, "1girl", tmp_path / "x.png"))
    assert ok is False
    assert calls["attempts"] == ranker.MAX_ATTEMPTS
    assert len(no_sleep) == ranker.MAX_ATTEMPTS - 1


# ── A4: fuller history and side bias ─────────────────────────────────

def _rec(winner, **extra):
    return ranker.ComparisonRecord(timestamp=1.0, artists_a=["alpha"], artists_b=["bravo"], winner=winner,
                                   image_a_path="a.png", image_b_path="b.png", **extra)


def test_history_record_keeps_full_context(tmp_path):
    h = ranker.ComparisonHistory(tmp_path / "h.json")
    h.add_record(_rec("A", prompt_a="1girl, artist: alpha", prompt_b="1girl, artist: bravo",
                      base_prompt="1girl, {artist_placeholder}", negative_prompt="lowres",
                      quality_toggle=True, uc_preset=0, seed_a=123, seed_b=123, model_id="nai-diffusion-5-full"))
    saved = json.loads((tmp_path / "h.json").read_text())[0]
    for k in ("prompt_a", "prompt_b", "base_prompt", "negative_prompt", "quality_toggle", "uc_preset", "seed_a", "seed_b", "model_id"):
        assert k in saved, k
    assert saved["seed_a"] == 123 and saved["model_id"] == "nai-diffusion-5-full" and saved["winner"] == "A"


def test_legacy_records_still_work(tmp_path):
    legacy = [{"timestamp": 1, "artists_a": ["alpha"], "artists_b": ["bravo"], "winner": "A", "image_a_path": "", "image_b_path": ""}]
    (tmp_path / "h.json").write_text(json.dumps(legacy))
    h = ranker.ComparisonHistory(tmp_path / "h.json")
    assert h.get_artist_stats()["alpha"]["wins"] == 1
    assert h.get_side_bias()["decided"] == 1


def test_side_bias_counts_only_decided_rounds(tmp_path):
    h = ranker.ComparisonHistory(tmp_path / "h.json")
    for w in ["A", "A", "B", "A", "draw"]:  # anything that is not A or B is not a side pick
        h.add_record(_rec(w))
    assert h.get_side_bias() == {"a_wins": 3, "b_wins": 1, "decided": 4, "a_rate": 0.75}


def test_side_bias_text():
    assert ranker.format_side_bias({"a_wins": 3, "b_wins": 1, "decided": 4, "a_rate": 0.75}) == \
        "**Side bias:** A picked 75% of 4 decided rounds"
    assert ranker.format_side_bias({"a_wins": 0, "b_wins": 0, "decided": 0, "a_rate": None}) == \
        "**Side bias:** no decided rounds yet"


# ── A2: background prefetch of the next pair ─────────────────────────

def _slow_gen(delay, log, active):
    def gen(settings):
        with active["lock"]:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        try:
            time.sleep(delay)
            log.append(settings)
            return ("pair-for", settings)
        finally:
            with active["lock"]:
                active["n"] -= 1
    return gen


def _tracker():
    return [], {"n": 0, "max": 0, "lock": threading.Lock()}


S1 = ("1girl, {artist_placeholder}", None, True, 0)
S2 = ("1boy, {artist_placeholder}", "lowres", True, 0)


def test_prefetched_pair_is_served_when_settings_match():
    log, active = _tracker()
    p = ranker.PairPrefetcher(_slow_gen(0.05, log, active))
    s = ranker.PairSettings(*S1)
    p.start(s)
    assert p.take(s) == ("pair-for", s)
    assert log == [s]
    assert p.take(s) is None, "a served pair is consumed"


def test_prefetch_for_other_settings_is_discarded():
    log, active = _tracker()
    p = ranker.PairPrefetcher(_slow_gen(0.02, log, active))
    s1, s2 = ranker.PairSettings(*S1), ranker.PairSettings(*S2)
    p.start(s1)
    time.sleep(0.15)
    assert p.take(s2) is None
    assert p.take(s1) is None, "the stale pair must not be served later"


def test_take_waits_for_matching_prefetch_in_flight():
    log, active = _tracker()
    p = ranker.PairPrefetcher(_slow_gen(0.2, log, active))
    s = ranker.PairSettings(*S1)
    p.start(s)
    t0 = time.time()
    assert p.take(s) == ("pair-for", s)
    assert time.time() - t0 >= 0.15
    assert log == [s]


def test_start_does_not_duplicate_in_flight_or_ready_work():
    log, active = _tracker()
    p = ranker.PairPrefetcher(_slow_gen(0.05, log, active))
    s = ranker.PairSettings(*S1)
    p.start(s); p.start(s)
    assert p.take(s) is not None and log == [s]
    p.start(s)
    time.sleep(0.15)
    p.start(s)
    time.sleep(0.15)
    assert log == [s, s]


def test_generation_never_overlaps():
    log, active = _tracker()
    p = ranker.PairPrefetcher(_slow_gen(0.15, log, active))
    s1, s2 = ranker.PairSettings(*S1), ranker.PairSettings(*S2)
    p.start(s1)
    assert p.generate_now(s2) == ("pair-for", s2)
    p.take(s1)
    assert active["max"] == 1, "NovelAI allows one generation per account at a time"
    assert sorted(map(str, log)) == sorted(map(str, [s1, s2]))
