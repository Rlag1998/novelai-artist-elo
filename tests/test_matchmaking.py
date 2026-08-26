"""
Fail-first tests for skill-based pair selection: side B is chosen from sampled
candidate combinations by TrueSkill match quality, with an exploration share
that stays fully random.
"""
import random

import pytest

import config
import artist_elo_ranker as ranker
from skill import SkillSystem


@pytest.fixture
def manager(tmp_path):
    f = tmp_path / "tags.txt"
    f.write_text("anchor\nnear\nfar1\nfar2\nfar3\nfar4\n")
    return ranker.ArtistTagManager(f)  # no pool: candidates come from all artists


def _skill():
    s = SkillSystem()
    s.set("anchor", 25, 1)
    s.set("near", 25, 8)
    for f in ("far1", "far2", "far3", "far4"):
        s.set(f, 60, 1)
    return s


def test_matchmaking_defaults():
    assert config.MATCHMAKING == "skill"
    assert 0 < config.EXPLORE_RATE < 1
    assert config.MATCH_CANDIDATES >= 10


def test_skill_mode_picks_the_most_informative_opponent(manager, monkeypatch):
    manager.attach_skill(_skill(), mode="skill", candidates=200, explore_rate=0.0)
    for _ in range(5):
        assert manager.get_opponent(["anchor"], min_artists=1, max_artists=1) == ["near"]


def test_opponent_is_never_the_same_combination(manager):
    manager.attach_skill(_skill(), mode="skill", candidates=50, explore_rate=0.0)
    for _ in range(20):
        assert set(manager.get_opponent(["near"], min_artists=1, max_artists=1)) != {"near"}


def test_random_mode_ignores_skill(manager):
    manager.attach_skill(_skill(), mode="random", candidates=200, explore_rate=0.0)
    random.seed(7)
    picks = {tuple(manager.get_opponent(["anchor"], min_artists=1, max_artists=1)) for _ in range(60)}
    assert len(picks) > 1, "random mode must not always pick the best-quality opponent"


def test_exploration_share_bypasses_scoring(manager, monkeypatch):
    manager.attach_skill(_skill(), mode="skill", candidates=50, explore_rate=0.5)
    monkeypatch.setattr(ranker.random, "random", lambda: 0.1)          # below explore_rate → explore
    monkeypatch.setattr(manager, "get_random_combination", lambda *a, **k: ["far3"])
    assert manager.get_opponent(["anchor"]) == ["far3"]


def test_pair_generation_uses_the_matchmaker(tmp_path, monkeypatch, manager):
    import asyncio
    manager.attach_skill(_skill(), mode="skill", candidates=200, explore_rate=0.0)
    original = manager.get_random_combination
    calls = {"n": 0}

    def first_anchor_then_solos(*a, **k):
        calls["n"] += 1
        return ["anchor"] if calls["n"] == 1 else original(1, 1)

    monkeypatch.setattr(manager, "get_random_combination", first_anchor_then_solos)
    payloads = []

    async def fake_request(self, session=None, **kwargs):
        payloads.append(1)
        from conftest import FakeResp
        return FakeResp()

    monkeypatch.setattr(ranker.GenerateImageInfer, "request", fake_request)
    pair = asyncio.run(ranker.generate_comparison_pair("1girl, {artist_placeholder}", manager, None, tmp_path))
    assert pair.artists_a == ["anchor"] and pair.artists_b == ["near"]
