"""
Fail-first tests for draws ("Same" button): zero-sum ELO, TrueSkill draw,
history and display, plus skill bootstrap and undo through the real app.
"""
import json

import pytest

import artist_elo_ranker as ranker
from skill import DEFAULT_SIGMA


def _elo(**ratings):
    e = ranker.ELOSystem()
    e.ratings.update(ratings)
    return e


def test_draw_between_equals_changes_nothing_but_counts():
    e = _elo(a=1500.0, b=1500.0)
    e.update_ratings(["a"], ["b"], draw=True)
    assert e.ratings["a"] == 1500.0 and e.ratings["b"] == 1500.0
    assert e.comparison_count == 1 and e.artist_comparisons == {"a": 1, "b": 1}


def test_uneven_draw_is_zero_sum_and_favours_the_underdog():
    e = _elo(hi=1600.0, lo=1400.0)
    e.update_ratings(["hi"], ["lo"], draw=True)
    assert e.ratings["hi"] < 1600.0 < 1600.0 + (1400.0 - e.ratings["lo"]) * -1 + 1e-9
    assert e.ratings["lo"] > 1400.0
    assert (e.ratings["hi"] - 1600.0) + (e.ratings["lo"] - 1400.0) == pytest.approx(0.0, abs=1e-9)


def test_team_draw_is_zero_sum():
    e = _elo(a1=1700.0, a2=1500.0, b1=1450.0)
    before = dict(e.ratings)
    e.update_ratings(["a1", "a2"], ["b1"], draw=True)
    assert sum(e.ratings[k] - before[k] for k in before) == pytest.approx(0.0, abs=1e-9)
    assert e.ratings["b1"] > 1450.0, "the lower-rated side gains from a draw"


def test_win_path_is_unchanged():
    e = _elo(a=1500.0, b=1500.0)
    e.update_ratings(["a"], ["b"])
    assert e.ratings["a"] == pytest.approx(1516.0) and e.ratings["b"] == pytest.approx(1484.0)


def test_history_counts_draws(tmp_path):
    h = ranker.ComparisonHistory(tmp_path / "h.json")
    for w in ("A", "draw"):
        h.add_record(ranker.ComparisonRecord(timestamp=1.0, artists_a=["a"], artists_b=["b"], winner=w,
                                             image_a_path="", image_b_path=""))
    stats = h.get_artist_stats()
    assert stats["a"] == {**stats["a"], "rounds": 2, "wins": 1, "draws": 1}
    assert stats["b"]["draws"] == 1 and stats["b"]["wins"] == 0


# ── through the real app ─────────────────────────────────────────────

def test_draw_flow_updates_elo_skill_history_and_undoes(app):
    app.generate_new_comparison("", "", True, 0)
    a, b = app.current_pair.artists_a, app.current_pair.artists_b
    snap = app.skill.snapshot(a + b)

    out = app.pick_winner("draw")
    assert "draw" in out[0].lower()
    rec = json.loads((app._tmp / "history.json").read_text())[-1]
    assert rec["winner"] == "draw"
    assert app.elo_system.comparison_count == 1
    assert any(app.skill.sigma_of(x) < DEFAULT_SIGMA for x in a + b), "TrueSkill learned from the draw"
    assert "drew with" in app.format_recent_history()

    app.undo_last_selection()
    assert app.elo_system.comparison_count == 0
    for x in a + b:
        assert (app.skill.mu_of(x), app.skill.sigma_of(x)) == pytest.approx(snap[x])
    assert json.loads((app._tmp / "history.json").read_text()) == []


def test_pick_updates_skill_and_display(app):
    app.generate_new_comparison("", "", True, 0)
    winners = app.current_pair.artists_a
    app.pick_winner("A")
    assert all(app.skill.mu_of(w) > 25 for w in winners)
    board = app.format_top_artists_display()
    assert "σ" in board and "settled" in board.lower()
    csv = app.export_leaderboard_csv()
    assert "Skill_Mu" in csv.splitlines()[0] and "Skill_Sigma" in csv.splitlines()[0]


def test_skill_is_bootstrapped_from_existing_history(app_env):
    history = [
        {"timestamp": 1, "artists_a": ["artist1"], "artists_b": ["artist2"], "winner": "A", "image_a_path": "", "image_b_path": ""},
        {"timestamp": 2, "artists_a": ["artist2"], "artists_b": ["artist3"], "winner": "B", "image_a_path": "", "image_b_path": ""},
    ]
    a = app_env(history=history)
    assert a.skill.mu_of("artist1") > 25 > a.skill.mu_of("artist2")
    assert (a._tmp / "skill.json").exists(), "bootstrapped skills are persisted"
