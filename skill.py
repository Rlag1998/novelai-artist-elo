"""
Uncertainty-aware ratings for artist tags, kept beside the ELO.

The ELO remains the canonical, zero-sum leaderboard. This layer adds a
per-artist mean (mu) and uncertainty (sigma) using TrueSkill. It drives pair
selection (match the most informative opponents) and the confidence shown on
the leaderboard. Teams of 1 to 3 artists use partial-play weights of 1/n, so a
trio has no structural advantage over a solo.
"""
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import trueskill

from config import SETTLED_SIGMA

DEFAULT_MU = 25.0
DEFAULT_SIGMA = 25.0 / 3
DRAW_PROBABILITY = 0.10


class SkillSystem:
    """TrueSkill ratings for artists, with team weights of 1/n."""

    def __init__(self, filepath: Optional[Path] = None, draw_probability: float = DRAW_PROBABILITY):
        self.filepath = Path(filepath) if filepath else None
        self.env = trueskill.TrueSkill(mu=DEFAULT_MU, sigma=DEFAULT_SIGMA, draw_probability=draw_probability)
        self.ratings: Dict[str, trueskill.Rating] = {}

    # ── persistence ──────────────────────────────────────────────────

    @classmethod
    def load(cls, filepath: Path) -> "SkillSystem":
        system = cls(filepath)
        fp = Path(filepath)
        if fp.exists():
            data = json.loads(fp.read_text(encoding="utf-8"))
            for artist, (mu, sigma) in data.get("ratings", {}).items():
                system.ratings[artist] = system.env.create_rating(mu, sigma)
        return system

    def save(self, filepath: Optional[Path] = None):
        fp = Path(filepath or self.filepath)
        payload = {"ratings": {a: [r.mu, r.sigma] for a, r in self.ratings.items()}}
        fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ── lookups ──────────────────────────────────────────────────────

    def get(self, artist: str) -> trueskill.Rating:
        return self.ratings.get(artist) or self.env.create_rating()

    def mu_of(self, artist: str) -> float:
        return self.get(artist).mu

    def sigma_of(self, artist: str) -> float:
        return self.get(artist).sigma

    def set(self, artist: str, mu: float, sigma: float):
        self.ratings[artist] = self.env.create_rating(mu, sigma)

    def snapshot(self, artists: Iterable[str]) -> Dict[str, Tuple[float, float]]:
        return {a: (self.get(a).mu, self.get(a).sigma) for a in artists}

    def restore(self, snapshot: Dict[str, Tuple[float, float]]):
        for artist, (mu, sigma) in snapshot.items():
            self.set(artist, mu, sigma)

    def settled(self, artist: str, threshold: float = SETTLED_SIGMA) -> bool:
        """An artist counts as settled once its uncertainty is small."""
        return artist in self.ratings and self.get(artist).sigma <= threshold

    # ── team maths ───────────────────────────────────────────────────

    @staticmethod
    def _weights(team: List[str]) -> Tuple[float, ...]:
        return tuple(1.0 / len(team) for _ in team)

    def _groups(self, team_a: List[str], team_b: List[str]):
        groups = [[self.get(a) for a in team_a], [self.get(b) for b in team_b]]
        return groups, [self._weights(team_a), self._weights(team_b)]

    def team_mu(self, team: List[str]) -> float:
        return sum(self.get(a).mu for a in team) / len(team) if team else DEFAULT_MU

    def win_probability(self, team_a: List[str], team_b: List[str]) -> float:
        """P(team A beats team B) under the model, with 1/n weights."""
        (ga, gb), (wa, wb) = self._groups(team_a, team_b)
        delta = sum(w * r.mu for w, r in zip(wa, ga)) - sum(w * r.mu for w, r in zip(wb, gb))
        var = (sum((w * r.sigma) ** 2 for w, r in zip(wa, ga))
               + sum((w * r.sigma) ** 2 for w, r in zip(wb, gb))
               + 2 * self.env.beta ** 2)
        return 0.5 * (1 + math.erf(delta / math.sqrt(2 * var)))

    def quality(self, team_a: List[str], team_b: List[str]) -> float:
        """TrueSkill match quality: highest when the outcome is most uncertain,
        which is when a comparison teaches the most."""
        groups, weights = self._groups(team_a, team_b)
        return self.env.quality(groups, weights=weights)

    # ── updates ──────────────────────────────────────────────────────

    def update(self, team_a: List[str], team_b: List[str], outcome: str) -> bool:
        """Apply one result. outcome is "A", "B" or "draw". Artists on both sides
        are neutral and excluded. Returns False when nothing could be learned."""
        overlap = set(team_a) & set(team_b)
        a = [x for x in team_a if x not in overlap]
        b = [x for x in team_b if x not in overlap]
        if not a or not b or outcome not in ("A", "B", "draw"):
            return False
        ranks = {"A": [0, 1], "B": [1, 0], "draw": [0, 0]}[outcome]
        groups, weights = self._groups(a, b)
        new_a, new_b = self.env.rate(groups, ranks=ranks, weights=weights)
        for artist, rating in zip(a, new_a):
            self.ratings[artist] = rating
        for artist, rating in zip(b, new_b):
            self.ratings[artist] = rating
        return True

    def rebuild_from_records(self, records: Iterable[dict]) -> int:
        """Recompute every rating from a comparison history. Returns the number
        of records that were usable (winner A, B or draw)."""
        self.ratings = {}
        applied = 0
        for rec in records:
            if self.update(rec.get("artists_a", []), rec.get("artists_b", []), rec.get("winner", "")):
                applied += 1
        return applied
