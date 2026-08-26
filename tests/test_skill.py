"""
Fail-first tests for uncertainty-aware ratings: a TrueSkill layer kept beside
the ELO. Teams of 1 to 3 artists use partial-play weights of 1/n so team size
carries no structural advantage.
"""
import json
import math

import pytest

import config
from skill import SkillSystem, DEFAULT_MU, DEFAULT_SIGMA


def test_defaults_and_unknown_artists():
    s = SkillSystem()
    r = s.get("nobody")
    assert (r.mu, round(r.sigma, 3)) == (DEFAULT_MU, round(DEFAULT_SIGMA, 3))
    assert s.mu_of("nobody") == DEFAULT_MU and s.sigma_of("nobody") == pytest.approx(DEFAULT_SIGMA)


def test_win_moves_means_apart_and_shrinks_sigma():
    s = SkillSystem()
    assert s.update(["a"], ["b"], "A") is True
    assert s.mu_of("a") > DEFAULT_MU > s.mu_of("b")
    assert s.sigma_of("a") < DEFAULT_SIGMA and s.sigma_of("b") < DEFAULT_SIGMA


def test_draw_pulls_means_together_and_shrinks_sigma():
    s = SkillSystem()
    s.set("hi", 30, 5); s.set("lo", 20, 5)
    s.update(["hi"], ["lo"], "draw")
    assert 20 < s.mu_of("lo") < s.mu_of("hi") < 30
    assert s.sigma_of("hi") < 5 and s.sigma_of("lo") < 5


def test_team_size_carries_no_advantage():
    s = SkillSystem()
    assert s.win_probability(["solo"], ["t1", "t2", "t3"]) == pytest.approx(0.5)
    assert s.quality(["solo"], ["t1", "t2", "t3"]) > 0.4


def test_team_outcome_is_shared_by_members():
    s = SkillSystem()
    s.update(["solo"], ["t1", "t2"], "B")
    assert s.mu_of("t1") == pytest.approx(s.mu_of("t2")) and s.mu_of("t1") > DEFAULT_MU
    assert s.mu_of("solo") < DEFAULT_MU


def test_overlapping_artist_is_neutral():
    s = SkillSystem()
    s.update(["x", "a"], ["x", "b"], "A")
    assert s.mu_of("x") == DEFAULT_MU, "an artist on both sides cannot win or lose"
    assert s.mu_of("a") > DEFAULT_MU > s.mu_of("b")


def test_fully_overlapping_teams_are_skipped():
    s = SkillSystem()
    assert s.update(["x"], ["x"], "A") is False
    assert s.ratings == {}


def test_snapshot_restore_supports_undo():
    s = SkillSystem()
    snap = s.snapshot(["a", "b"])
    s.update(["a"], ["b"], "A")
    s.restore(snap)
    assert s.mu_of("a") == DEFAULT_MU and s.sigma_of("b") == pytest.approx(DEFAULT_SIGMA)


def test_persistence_roundtrip(tmp_path):
    f = tmp_path / "skill.json"
    s = SkillSystem(f)
    s.update(["a"], ["b"], "A")
    s.save()
    again = SkillSystem.load(f)
    assert again.mu_of("a") == pytest.approx(s.mu_of("a")) and again.sigma_of("b") == pytest.approx(s.sigma_of("b"))
    assert SkillSystem.load(tmp_path / "missing.json").ratings == {}


def test_rebuild_from_history_matches_sequential_updates():
    records = [
        {"artists_a": ["a"], "artists_b": ["b"], "winner": "A"},
        {"artists_a": ["b", "c"], "artists_b": ["a"], "winner": "B"},
        {"artists_a": ["c"], "artists_b": ["a"], "winner": "draw"},
        {"artists_a": ["c"], "artists_b": ["b"], "winner": "skip"},   # ignored
    ]
    seq = SkillSystem()
    seq.update(["a"], ["b"], "A"); seq.update(["b", "c"], ["a"], "B"); seq.update(["c"], ["a"], "draw")
    rebuilt = SkillSystem()
    assert rebuilt.rebuild_from_records(records) == 3
    for who in "abc":
        assert rebuilt.mu_of(who) == pytest.approx(seq.mu_of(who))
        assert rebuilt.sigma_of(who) == pytest.approx(seq.sigma_of(who))


def test_settled_threshold_is_configured():
    s = SkillSystem()
    assert config.SETTLED_SIGMA > 0
    assert s.settled("fresh") is False
    s.set("veteran", 30, config.SETTLED_SIGMA - 0.1)
    assert s.settled("veteran") is True
