#!/usr/bin/env python3
"""
Offline comparison of pairing criteria using the real selection code.

Simulates a population with known latent quality and a noisy judge, runs rounds
through ArtistTagManager.get_opponent(), and reports how well the resulting ELO
recovers the true ranking. No network, no data files touched.

Usage:  python3 scripts/simulate_pairing.py [--rounds 300] [--artists 60] [--seeds 10]
"""
import argparse
import contextlib
import io
import math
import random
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with contextlib.redirect_stdout(io.StringIO()):
    import artist_elo_ranker as ranker
    from skill import SkillSystem


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


def top_k(truth, est, k=10):
    t = set(sorted(truth, key=truth.get, reverse=True)[:k])
    e = set(sorted(est, key=est.get, reverse=True)[:k])
    return len(t & e) / k


CRITERIA = {
    "random (default)": None,
    "quality": lambda m, a, c: m.skill_system.quality(a, c),
    "sigma*quality": lambda m, a, c: (sum(m.skill_system.sigma_of(x) for x in c) / len(c)) * m.skill_system.quality(a, c),
    "least-seen": lambda m, a, c: -sum(m.elo_system.get_artist_comparison_count(x) for x in c) / len(c),
}


def simulate(criterion, seed, noise, rounds, artists):
    random.seed(seed)
    names = [f"a{i}" for i in range(artists)]
    truth = {n: random.gauss(0, 1) for n in names}
    tmp = Path(tempfile.mkdtemp())
    (tmp / "tags.txt").write_text("\n".join(names))
    with contextlib.redirect_stdout(io.StringIO()):
        manager = ranker.ArtistTagManager(tmp / "tags.txt")
    elo, skill = ranker.ELOSystem(), SkillSystem()
    manager.elo_system = elo
    score = CRITERIA[criterion]
    if score is None:
        manager.attach_skill(skill, mode="random")
    else:
        manager.attach_skill(skill, mode="skill", candidates=30, explore_rate=0.2)
        manager.match_score = lambda a, c: score(manager, a, c)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(rounds):
            a = manager.get_random_combination()
            b = manager.get_opponent(a)
            ta = sum(truth[x] for x in a) / len(a)
            tb = sum(truth[x] for x in b) / len(b)
            if random.random() < 1 / (1 + math.exp(-(ta - tb) * noise)):
                elo.update_ratings(a, b); skill.update(a, b, "A")
            else:
                elo.update_ratings(b, a); skill.update(a, b, "B")
    est = {n: elo.get_rating(n) for n in names}
    return spearman([truth[n] for n in names], [est[n] for n in names]), top_k(truth, est)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--artists", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    print(f"{'judge':>12s} {'criterion':18s} {'ELO rank corr':>13s} {'top-10 hit':>10s}   "
          f"({args.rounds} rounds, {args.artists} artists, {args.seeds} seeds)")
    for noise, label in ((1.0, "very noisy"), (1.7, "noisy"), (4.0, "consistent"), (8.0, "near-perfect")):
        for name in CRITERIA:
            res = [simulate(name, s, noise, args.rounds, args.artists) for s in range(args.seeds)]
            print(f"{label:>12s} {name:18s} {statistics.mean(r[0] for r in res):13.3f} {statistics.mean(r[1] for r in res):10.2f}")
        print()


if __name__ == "__main__":
    main()
