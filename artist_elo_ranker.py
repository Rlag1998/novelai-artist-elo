#!/usr/bin/env python3
"""
Artist ELO Ranking System for NovelAI Image Generation

A blind comparison system that generates images with random artist tag combinations
(1-3 artists) and allows users to pick their preferred image. Artists gain/lose ELO
based on the outcomes.
"""

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from pydantic import SecretStr

from novelai_python import GenerateImageInfer, ImageGenerateResp, ApiCredential
from novelai_python.sdk.ai.generate_image import Model, Sampler, UCPreset

# Import configuration
from config import (
    get_api_key,
    ARTIST_TAGS_FILE,
    ELO_RATINGS_FILE,
    COMPARISON_IMAGES_DIR,
    COMPARISON_HISTORY_FILE,
    ACTIVE_POOL_FILE,
    STEPS,
    IMG_WIDTH,
    IMG_HEIGHT,
    DEFAULT_ELO,
    K_FACTOR,
    ACTIVE_POOL_SIZE,
    NEW_ARTIST_PROBABILITY,
    LOSER_ROTATION_PROBABILITY,
    SERVER_HOST,
    SERVER_PORT,
    NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
)

# --------------------------------------------------------------------------------
# NovelAI Model Configuration
# --------------------------------------------------------------------------------

MODEL = Model.NAI_DIFFUSION_4_5_FULL
SAMPLER = Sampler.K_EULER_ANCESTRAL
UC_PRESET = UCPreset.TYPE0


# --------------------------------------------------------------------------------
# ELO Rating System
# --------------------------------------------------------------------------------

@dataclass
class ELOSystem:
    """Manages ELO ratings for artists."""
    ratings: dict = field(default_factory=dict)
    comparison_count: int = 0
    artist_comparisons: dict = field(default_factory=dict)  # Track per-artist comparisons

    @classmethod
    def load(cls, filepath: Path) -> "ELOSystem":
        """Load ELO ratings from file."""
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                system = cls()
                system.ratings = data.get("ratings", {})
                system.comparison_count = data.get("comparison_count", 0)
                system.artist_comparisons = data.get("artist_comparisons", {})
                return system
        return cls()

    def save(self, filepath: Path):
        """Save ELO ratings to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "ratings": self.ratings,
                "comparison_count": self.comparison_count,
                "artist_comparisons": self.artist_comparisons
            }, f, indent=2)

    def get_rating(self, artist: str) -> float:
        """Get ELO rating for an artist, defaulting to DEFAULT_ELO."""
        return self.ratings.get(artist, DEFAULT_ELO)

    def get_combined_rating(self, artists: List[str]) -> float:
        """Get average ELO rating for a combination of artists."""
        if not artists:
            return DEFAULT_ELO
        return sum(self.get_rating(a) for a in artists) / len(artists)

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, winners: List[str], losers: List[str]):
        """
        Update ELO ratings after a comparison.
        Uses INDIVIDUAL-based calculation: each artist's gain/loss is based on
        their own ELO vs the opposing team's average (not team vs team).

        Scaled to maintain zero-sum: total ELO gained = total ELO lost.
        """
        # Find overlapping artists (they're neutral - no ELO change)
        overlap = set(winners) & set(losers)
        actual_winners = [a for a in winners if a not in overlap]
        actual_losers = [a for a in losers if a not in overlap]

        if overlap:
            print(f"Overlap detected (neutral): {overlap}")

        if not actual_winners or not actual_losers:
            self.comparison_count += 1
            return

        # Get opposing team averages for individual calculations
        winner_team_avg = self.get_combined_rating(winners)
        loser_team_avg = self.get_combined_rating(losers)

        # Calculate raw changes for winners
        winner_changes = []
        for artist in actual_winners:
            current = self.get_rating(artist)
            expected = self.calculate_expected_score(current, loser_team_avg)
            change = K_FACTOR * (1 - expected)
            winner_changes.append((artist, change))

        # Calculate raw changes for losers
        loser_changes = []
        for artist in actual_losers:
            current = self.get_rating(artist)
            expected = self.calculate_expected_score(current, winner_team_avg)
            change = K_FACTOR * (0 - (1 - expected))
            loser_changes.append((artist, change))

        # Scale to maintain zero-sum
        total_winner_gain = sum(c for _, c in winner_changes)
        total_loser_loss = sum(c for _, c in loser_changes)  # negative

        # Scale loser losses so total loss = total gain
        if total_loser_loss != 0:
            scale_factor = -total_winner_gain / total_loser_loss
        else:
            scale_factor = 1.0

        # Apply changes
        for artist, change in winner_changes:
            self.ratings[artist] = self.get_rating(artist) + change
            self.artist_comparisons[artist] = self.artist_comparisons.get(artist, 0) + 1

        for artist, change in loser_changes:
            scaled_change = change * scale_factor
            self.ratings[artist] = self.get_rating(artist) + scaled_change
            self.artist_comparisons[artist] = self.artist_comparisons.get(artist, 0) + 1

        self.comparison_count += 1

    def get_top_artists(self, n: int = 50) -> List[Tuple[str, float, int]]:
        """Get top N artists by ELO rating with their comparison counts."""
        sorted_artists = sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        return [(artist, rating, self.get_artist_comparison_count(artist))
                for artist, rating in sorted_artists]

    def get_bottom_artists(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get bottom N artists by ELO rating."""
        sorted_artists = sorted(
            self.ratings.items(),
            key=lambda x: x[1]
        )[:n]
        return sorted_artists

    def get_artist_comparison_count(self, artist: str) -> int:
        """Get the number of comparisons an artist has participated in."""
        return self.artist_comparisons.get(artist, 0)


# --------------------------------------------------------------------------------
# Active Pool System
# --------------------------------------------------------------------------------

class ActivePool:
    """
    Manages a smaller active pool of artists for more meaningful comparisons.

    Strategy:
    - Maintain a pool of ~150 artists that get compared frequently
    - Winners stay in the pool (good artists get more comparisons)
    - Losers may get rotated out (with some probability)
    - Periodically introduce new random artists to discover hidden gems
    - Weight selection towards artists with fewer comparisons (need more data)
    """

    def __init__(self, all_artists: List[str], elo_system: ELOSystem,
                 pool_size: int = ACTIVE_POOL_SIZE, pool_file: Path = None):
        self.all_artists = all_artists
        self.elo_system = elo_system
        self.pool_size = pool_size
        self.pool_file = pool_file or ACTIVE_POOL_FILE
        self.pool: List[str] = []
        self.load()

    def load(self):
        """Load pool from file or initialize if not exists."""
        if self.pool_file.exists():
            with open(self.pool_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.pool = data.get("pool", [])
                # Validate pool members still exist in all_artists
                self.pool = [a for a in self.pool if a in self.all_artists]

        # Initialize or refill pool if needed
        if len(self.pool) < self.pool_size:
            self._refill_pool()

    def save(self):
        """Save pool to file."""
        with open(self.pool_file, "w", encoding="utf-8") as f:
            json.dump({"pool": self.pool}, f, indent=2)

    def _refill_pool(self):
        """Fill pool up to pool_size with random artists not already in pool."""
        available = [a for a in self.all_artists if a not in self.pool]
        needed = self.pool_size - len(self.pool)
        if needed > 0 and available:
            new_artists = random.sample(available, min(needed, len(available)))
            self.pool.extend(new_artists)
            print(f"Added {len(new_artists)} new artists to active pool. Pool size: {len(self.pool)}")
        self.save()

    def get_selection_weight(self, artist: str) -> float:
        """
        Calculate selection weight for an artist.
        Artists with fewer comparisons get higher weight (need more data).
        """
        comparisons = self.elo_system.get_artist_comparison_count(artist)
        # Inverse weight: fewer comparisons = higher weight
        # Add 1 to avoid division by zero, use sqrt to dampen the effect
        return 1.0 / (1.0 + (comparisons ** 0.5))

    def select_artist(self) -> str:
        """Select a single artist from the pool, weighted by need for comparisons."""
        if not self.pool:
            self._refill_pool()

        weights = [self.get_selection_weight(a) for a in self.pool]
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(self.pool)

        # Weighted random selection
        r = random.uniform(0, total_weight)
        cumulative = 0
        for artist, weight in zip(self.pool, weights):
            cumulative += weight
            if r <= cumulative:
                return artist
        return self.pool[-1]

    def select_combination(self, min_artists: int = 1, max_artists: int = 3) -> List[str]:
        """Select a combination of 1-3 artists from the pool."""
        if not self.pool:
            self._refill_pool()

        num_artists = random.randint(min_artists, max_artists)
        num_artists = min(num_artists, len(self.pool))

        selected = []
        pool_copy = self.pool.copy()

        for _ in range(num_artists):
            if not pool_copy:
                break
            weights = [self.get_selection_weight(a) for a in pool_copy]
            total_weight = sum(weights)
            if total_weight == 0:
                artist = random.choice(pool_copy)
            else:
                r = random.uniform(0, total_weight)
                cumulative = 0
                artist = pool_copy[-1]
                for a, w in zip(pool_copy, weights):
                    cumulative += w
                    if r <= cumulative:
                        artist = a
                        break
            selected.append(artist)
            pool_copy.remove(artist)

        # Shuffle to randomize tag order in prompt
        random.shuffle(selected)
        return selected

    def process_result(self, winners: List[str], losers: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float, bool]]]:
        """
        Process comparison result to update the pool.
        Uses weighted removal from entire pool (not just losers).

        Returns: (rotated_out, rotated_in) where each is list of (artist, elo, [is_returning])
        """
        rotated_out = []  # [(artist, elo), ...]
        rotated_in = []   # [(artist, elo, is_returning), ...]

        # Determine probabilities based on pool size
        pool_diff = len(self.pool) - self.pool_size
        if pool_diff > 10:  # Pool too big
            removal_prob = 0.3
            addition_prob = 0.05
        elif pool_diff < -10:  # Pool too small
            removal_prob = 0.05
            addition_prob = 0.3
        else:  # Around target
            removal_prob = 0.15
            addition_prob = 0.15

        # Weighted removal from entire pool (not just losers)
        # Weight = confidence(matches) * underperformance(relative to pool max)
        if random.random() < removal_prob and len(self.pool) > self.pool_size // 2:
            pool_elos = [self.elo_system.get_rating(a) for a in self.pool]
            pool_max_elo = max(pool_elos)

            removal_weights = []
            for artist in self.pool:
                matches = self.elo_system.get_artist_comparison_count(artist)
                elo = self.elo_system.get_rating(artist)

                # Confidence: require 5+ matches before eligible for removal
                confidence = 1.0 if matches >= 5 else 0.0

                # Underperformance: relative to pool's best performer (squared)
                # Squared so worst performers are MUCH more likely to be removed
                # Mirrors the squared addition weighting for symmetry
                underperformance = ((pool_max_elo - elo) / 100.0) ** 2

                weight = confidence * underperformance
                removal_weights.append(weight)

            total_weight = sum(removal_weights)
            if total_weight > 0:
                r = random.uniform(0, total_weight)
                cumulative = 0
                for artist, weight in zip(self.pool, removal_weights):
                    cumulative += weight
                    if r <= cumulative:
                        elo = self.elo_system.get_rating(artist)
                        self.pool.remove(artist)
                        rotated_out.append((artist, elo))
                        print(f"Rotated out: {artist} (ELO: {elo:.0f})")
                        break

        # Maybe introduce new artist, weighted by ELO (squared for stronger preference)
        # Higher ELO = much higher chance of being added back
        if random.random() < addition_prob:
            available = [a for a in self.all_artists if a not in self.pool]
            if available:
                elos = [self.elo_system.get_rating(a) for a in available]
                min_elo = min(elos)
                # Square the weight difference for stronger high-ELO preference
                # 1700 vs 1300: old=(500 vs 100)=5x, new=(500^2 vs 100^2)=25x
                weights = [((e - min_elo + 100) ** 2) for e in elos]
                total = sum(weights)
                r = random.uniform(0, total)
                cumulative = 0
                new_artist = available[-1]
                for a, w in zip(available, weights):
                    cumulative += w
                    if r <= cumulative:
                        new_artist = a
                        break
                self.pool.append(new_artist)
                elo = self.elo_system.get_rating(new_artist)
                is_returning = new_artist in self.elo_system.ratings
                rotated_in.append((new_artist, elo, is_returning))
                status = f"returning, ELO: {elo:.0f}" if is_returning else "fresh"
                print(f"Rotated in: {new_artist} ({status})")

        # Hard cap: if pool exceeds target + 20, force remove lowest performers
        # This ensures pool size stays bounded
        max_pool_size = self.pool_size + 20
        while len(self.pool) > max_pool_size:
            # Find artists with enough matches to judge (confidence >= 1)
            candidates = [(a, self.elo_system.get_rating(a))
                          for a in self.pool
                          if self.elo_system.get_artist_comparison_count(a) >= 5]
            if candidates:
                # Remove lowest ELO among confident artists
                worst = min(candidates, key=lambda x: x[1])
                self.pool.remove(worst[0])
                rotated_out.append((worst[0], worst[1]))
                print(f"Hard cap removal: {worst[0]} (ELO: {worst[1]:.0f})")
            else:
                # No confident artists, remove lowest ELO anyway
                worst = min(self.pool, key=lambda a: self.elo_system.get_rating(a))
                elo = self.elo_system.get_rating(worst)
                self.pool.remove(worst)
                rotated_out.append((worst, elo))
                print(f"Hard cap removal (low confidence): {worst} (ELO: {elo:.0f})")

        # Ensure pool doesn't get too small
        if len(self.pool) < self.pool_size // 2:
            self._refill_pool()

        self.save()
        return rotated_out, rotated_in

    def restore_artists(self, artists: List[str]):
        """Restore artists to the pool (for undo)."""
        for artist in artists:
            if artist not in self.pool and artist in self.all_artists:
                self.pool.append(artist)
        self.save()

    def get_pool_stats(self) -> dict:
        """Get statistics about the current pool."""
        if not self.pool:
            return {"size": 0, "avg_comparisons": 0, "avg_elo": DEFAULT_ELO,
                    "at_risk": [], "newcomers": 0, "safe": 0, "total_artists": 0}

        comparisons = [self.elo_system.get_artist_comparison_count(a) for a in self.pool]
        elos = [self.elo_system.get_rating(a) for a in self.pool]
        pool_max_elo = max(elos) if elos else DEFAULT_ELO
        pool_avg_elo = sum(elos) / len(elos) if elos else DEFAULT_ELO

        # Categorize artists relative to pool
        at_risk = []  # Lower ELO in pool + enough matches
        newcomers = 0  # < 5 matches (protected)
        safe = 0  # Top performers (above pool average)

        for artist in self.pool:
            matches = self.elo_system.get_artist_comparison_count(artist)
            elo = self.elo_system.get_rating(artist)

            if matches < 5:
                newcomers += 1
            elif elo >= pool_avg_elo:
                safe += 1
            else:
                # At risk: has enough matches AND below pool average
                # Calculate removal weight for sorting (relative to pool max)
                confidence = min(1.0, matches / 5.0)
                underperformance = (pool_max_elo - elo) / 100.0
                weight = confidence * underperformance
                at_risk.append((artist, elo, matches, weight))

        # Sort at_risk by weight (most likely to be removed first)
        at_risk.sort(key=lambda x: x[3], reverse=True)

        return {
            "size": len(self.pool),
            "avg_comparisons": sum(comparisons) / len(comparisons),
            "min_comparisons": min(comparisons),
            "max_comparisons": max(comparisons),
            "avg_elo": sum(elos) / len(elos),
            "total_artists": len(self.all_artists),
            "at_risk": at_risk[:10],  # Top 10 most at risk
            "at_risk_count": len(at_risk),
            "newcomers": newcomers,
            "safe": safe,
        }


# --------------------------------------------------------------------------------
# Artist Tag Management
# --------------------------------------------------------------------------------

class ArtistTagManager:
    """Manages loading and selecting artist tags."""

    def __init__(self, tags_file: Path, elo_system: ELOSystem = None):
        self.tags_file = tags_file
        self.artists: List[str] = []
        self.elo_system = elo_system
        self.active_pool: Optional[ActivePool] = None
        self.load_artists()

    def load_artists(self):
        """Load artist tags from file."""
        if self.tags_file.exists():
            with open(self.tags_file, "r", encoding="utf-8") as f:
                self.artists = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(self.artists)} artist tags")
        else:
            print(f"Warning: Artist tags file not found: {self.tags_file}")
            self.artists = []

    def initialize_pool(self, elo_system: ELOSystem):
        """Initialize the active pool with the ELO system."""
        self.elo_system = elo_system
        self.active_pool = ActivePool(self.artists, elo_system)

    def get_random_combination(self, min_artists: int = 1, max_artists: int = 3) -> List[str]:
        """Get a random combination of 1-3 artists, using active pool if available."""
        if self.active_pool:
            return self.active_pool.select_combination(min_artists, max_artists)

        # Fallback to pure random if no pool
        if not self.artists:
            return []
        num_artists = random.randint(min_artists, max_artists)
        return random.sample(self.artists, min(num_artists, len(self.artists)))

    def process_result(self, winners: List[str], losers: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float, bool]]]:
        """Process comparison result to update the active pool. Returns (rotated_out, rotated_in)."""
        if self.active_pool:
            return self.active_pool.process_result(winners, losers)
        return [], []

    def restore_artists(self, artists: List[str]):
        """Restore artists to the pool (for undo)."""
        if self.active_pool:
            self.active_pool.restore_artists(artists)

    def get_pool_stats(self) -> dict:
        """Get active pool statistics."""
        if self.active_pool:
            return self.active_pool.get_pool_stats()
        return {"size": 0, "total_artists": len(self.artists)}

    def format_artist_tags(self, artists: List[str]) -> str:
        """Format artist list as comma-separated artist tags."""
        return ", ".join(f"artist: {artist}" for artist in artists)


# --------------------------------------------------------------------------------
# Prompt Processing
# --------------------------------------------------------------------------------

def is_artist_tag(segment: str) -> bool:
    """
    Check if a segment is an artist tag.
    Handles: artist: name, artist:name, [artist: name], {artist: name:1.5}, etc.
    """
    stripped = segment.strip().lower()
    # Remove surrounding brackets
    stripped_no_brackets = stripped.strip("[]{}()")

    # Check for artist: or artist (space) patterns
    if stripped_no_brackets.startswith("artist:") or stripped_no_brackets.startswith("artist "):
        return True
    # Check for artist: anywhere (handles weighted like "artist:name:1.5")
    if "artist:" in stripped_no_brackets:
        return True
    # Check for "artist name" pattern at start
    if re.match(r"^\s*artist\s+\w", stripped_no_brackets):
        return True
    return False


def remove_artist_tags_with_position(prompt: str) -> tuple:
    """
    Remove existing artist tags from a prompt and return position of first one found.
    Returns: (cleaned_prompt, first_artist_index or -1 if none found)

    Handles:
    - artist: name, artist:name
    - [artist: name], {artist: name}, (artist: name)
    - artist:name:1.5 (weighted tags)
    """
    segments = prompt.split(",")
    filtered_segments = []
    first_artist_idx = -1

    for i, segment in enumerate(segments):
        if is_artist_tag(segment):
            if first_artist_idx == -1:
                first_artist_idx = len(filtered_segments)  # Position in filtered list
            continue
        filtered_segments.append(segment)

    return ",".join(filtered_segments), first_artist_idx


def remove_artist_tags(prompt: str) -> str:
    """Remove all artist tags from prompt, return cleaned prompt."""
    cleaned, _ = remove_artist_tags_with_position(prompt)
    return cleaned


def insert_artist_tags(prompt: str, artist_tags: str) -> str:
    """
    Insert artist tags into the prompt.
    If {artist_placeholder} exists, replace it.
    Otherwise, insert at position of first existing artist tag, or at end.
    """
    if "{artist_placeholder}" in prompt:
        return prompt.replace("{artist_placeholder}", artist_tags)

    # Remove any existing artist tags and get position of first one
    clean_prompt, first_artist_idx = remove_artist_tags_with_position(prompt)
    segments = clean_prompt.split(",")

    # If there was an existing artist tag, insert at that position
    if first_artist_idx >= 0 and first_artist_idx <= len(segments):
        segments.insert(first_artist_idx, f" {artist_tags}")
        return ",".join(segments)

    # No existing artist tags - always append at end
    return clean_prompt + ", " + artist_tags


# --------------------------------------------------------------------------------
# Image Generation
# --------------------------------------------------------------------------------

async def generate_image(
    session: ApiCredential,
    prompt: str,
    output_path: Path,
    negative_prompt: str = None,
    quality_toggle: bool = True,
    uc_preset: int = 0
) -> bool:
    """Generate a single image and save it."""
    try:
        # Map UC preset index to enum (-1 = None/disabled)
        uc_preset_map = {
            -1: None,  # Disabled
            0: UCPreset.TYPE0,
            1: UCPreset.TYPE1,
            2: UCPreset.TYPE2,
            3: UCPreset.TYPE3,
        }
        gen = GenerateImageInfer.build_generate(
            prompt=prompt,
            width=IMG_WIDTH,
            height=IMG_HEIGHT,
            model=MODEL,
            steps=STEPS,
            sampler=SAMPLER,
            negative_prompt=negative_prompt if negative_prompt else NEGATIVE_PROMPT,
            ucPreset=uc_preset_map.get(uc_preset, UCPreset.TYPE0),
            qualityToggle=quality_toggle,
            decrisp_mode=False,
            variety_boost=False,
        )

        resp = await gen.request(session=session)
        resp: ImageGenerateResp

        _, file_bytes = resp.files[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(file_bytes)

        return True
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


async def generate_comparison_pair(
    base_prompt: str,
    artist_manager: ArtistTagManager,
    session: ApiCredential,
    output_dir: Path,
    negative_prompt: str = None,
    quality_toggle: bool = True,
    uc_preset: int = 0
) -> Tuple[Optional[Path], Optional[Path], List[str], List[str]]:
    """
    Generate two images with different artist combinations.
    Returns: (image_a_path, image_b_path, artists_a, artists_b)
    """
    # Get two different artist combinations (overlap is allowed, handled in ELO calc)
    artists_a = artist_manager.get_random_combination()
    artists_b = artist_manager.get_random_combination()

    # Ensure they're not identical (but overlap is fine)
    max_attempts = 50
    attempts = 0
    while set(artists_a) == set(artists_b) and attempts < max_attempts:
        artists_b = artist_manager.get_random_combination()
        attempts += 1

    # Format artist tags
    tags_a = artist_manager.format_artist_tags(artists_a)
    tags_b = artist_manager.format_artist_tags(artists_b)

    # Create prompts
    prompt_a = insert_artist_tags(base_prompt, tags_a)
    prompt_b = insert_artist_tags(base_prompt, tags_b)

    # Generate unique filenames
    timestamp = int(time.time() * 1000)
    path_a = output_dir / f"compare_{timestamp}_a.png"
    path_b = output_dir / f"compare_{timestamp}_b.png"

    print(f"Generating image A with artists: {artists_a}")
    print(f"Prompt A: {prompt_a[:200]}...")
    success_a = await generate_image(session, prompt_a, path_a, negative_prompt, quality_toggle, uc_preset)

    print(f"Generating image B with artists: {artists_b}")
    print(f"Prompt B: {prompt_b[:200]}...")
    success_b = await generate_image(session, prompt_b, path_b, negative_prompt, quality_toggle, uc_preset)

    if success_a and success_b:
        return path_a, path_b, artists_a, artists_b
    return None, None, [], []


# --------------------------------------------------------------------------------
# Comparison History
# --------------------------------------------------------------------------------

@dataclass
class ComparisonRecord:
    """Record of a single comparison."""
    timestamp: float
    artists_a: List[str]
    artists_b: List[str]
    winner: str  # "A" or "B"
    image_a_path: str
    image_b_path: str


class ComparisonHistory:
    """Manages comparison history."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.records: List[dict] = []
        self.load()

    def load(self):
        """Load history from file."""
        if self.filepath.exists():
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    def save(self):
        """Save history to file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def add_record(self, record: ComparisonRecord):
        """Add a comparison record."""
        self.records.append({
            "timestamp": record.timestamp,
            "artists_a": record.artists_a,
            "artists_b": record.artists_b,
            "winner": record.winner,
            "image_a_path": record.image_a_path,
            "image_b_path": record.image_b_path
        })
        self.save()

    def get_artist_stats(self) -> dict:
        """
        Calculate stats for all artists from comparison history.
        Returns dict of artist -> {
            'rounds': total appearances,
            'wins': total wins,
            'solo': {'rounds': n, 'wins': n},
            'duo': {'rounds': n, 'wins': n},
            'trio': {'rounds': n, 'wins': n}
        }
        """
        stats = {}

        for record in self.records:
            artists_a = record.get("artists_a", [])
            artists_b = record.get("artists_b", [])
            winner = record.get("winner", "")

            # Determine group sizes
            size_a = len(artists_a)
            size_b = len(artists_b)

            # Process side A
            for artist in artists_a:
                if artist not in stats:
                    stats[artist] = {
                        'rounds': 0, 'wins': 0,
                        'solo': {'rounds': 0, 'wins': 0},
                        'duo': {'rounds': 0, 'wins': 0},
                        'trio': {'rounds': 0, 'wins': 0}
                    }
                stats[artist]['rounds'] += 1
                won = (winner == "A")
                if won:
                    stats[artist]['wins'] += 1

                # Track by group size
                size_key = {1: 'solo', 2: 'duo', 3: 'trio'}.get(size_a, 'trio')
                stats[artist][size_key]['rounds'] += 1
                if won:
                    stats[artist][size_key]['wins'] += 1

            # Process side B
            for artist in artists_b:
                if artist not in stats:
                    stats[artist] = {
                        'rounds': 0, 'wins': 0,
                        'solo': {'rounds': 0, 'wins': 0},
                        'duo': {'rounds': 0, 'wins': 0},
                        'trio': {'rounds': 0, 'wins': 0}
                    }
                stats[artist]['rounds'] += 1
                won = (winner == "B")
                if won:
                    stats[artist]['wins'] += 1

                # Track by group size
                size_key = {1: 'solo', 2: 'duo', 3: 'trio'}.get(size_b, 'trio')
                stats[artist][size_key]['rounds'] += 1
                if won:
                    stats[artist][size_key]['wins'] += 1

        return stats


# --------------------------------------------------------------------------------
# Gradio UI Application
# --------------------------------------------------------------------------------

@dataclass
class UndoState:
    """State needed to undo the last comparison."""
    winners: List[str]
    losers: List[str]
    old_ratings: dict  # artist -> old rating
    old_comparisons: dict  # artist -> old comparison count
    rotated_out: List[str]  # artists that were rotated out of pool
    # Previous comparison images (for restoring on undo)
    prev_image_a: Optional[str] = None
    prev_image_b: Optional[str] = None
    prev_artists_a: List[str] = field(default_factory=list)
    prev_artists_b: List[str] = field(default_factory=list)


class ArtistELORanker:
    """Main application class."""

    def __init__(self):
        self.elo_system = ELOSystem.load(ELO_RATINGS_FILE)
        self.artist_manager = ArtistTagManager(ARTIST_TAGS_FILE)
        # Initialize the active pool with ELO system
        self.artist_manager.initialize_pool(self.elo_system)
        self.history = ComparisonHistory(COMPARISON_HISTORY_FILE)
        self.session: Optional[ApiCredential] = None

        # Current comparison state
        self.current_image_a: Optional[Path] = None
        self.current_image_b: Optional[Path] = None
        self.current_artists_a: List[str] = []
        self.current_artists_b: List[str] = []
        self.current_prompt: str = DEFAULT_PROMPT

        # Undo state
        self.last_undo_state: Optional[UndoState] = None
        self.selection_made: bool = False  # Track if a selection was made for current pair

        # Rotation log: list of (type, artist, elo, extra_info) - most recent first
        # type: "out" or "in", extra_info: is_returning for "in"
        self.rotation_log: List[Tuple[str, str, float, Optional[bool]]] = []

        # Ensure output directory exists
        COMPARISON_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def get_session(self) -> ApiCredential:
        """Get or create API session."""
        if self.session is None:
            api_key = get_api_key()
            self.session = ApiCredential(api_token=SecretStr(api_key))
        return self.session

    def export_leaderboard_csv(self) -> str:
        """Export full leaderboard sorted by ELO as CSV with detailed stats."""
        sorted_artists = sorted(
            self.elo_system.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        artist_stats = self.history.get_artist_stats()

        lines = ["Rank,Artist,ELO,Comparisons,Wins,Losses,WinRate,Solo_Rounds,Solo_Wins,Solo_WR,Duo_Rounds,Duo_Wins,Duo_WR,Trio_Rounds,Trio_Wins,Trio_WR"]

        for rank, (artist, rating) in enumerate(sorted_artists, 1):
            comparisons = self.elo_system.get_artist_comparison_count(artist)
            stats = artist_stats.get(artist, {})

            rounds = stats.get('rounds', 0)
            wins = stats.get('wins', 0)
            losses = rounds - wins
            win_rate = (wins / rounds * 100) if rounds > 0 else 0

            solo = stats.get('solo', {'rounds': 0, 'wins': 0})
            duo = stats.get('duo', {'rounds': 0, 'wins': 0})
            trio = stats.get('trio', {'rounds': 0, 'wins': 0})

            solo_wr = (solo['wins'] / solo['rounds'] * 100) if solo['rounds'] > 0 else 0
            duo_wr = (duo['wins'] / duo['rounds'] * 100) if duo['rounds'] > 0 else 0
            trio_wr = (trio['wins'] / trio['rounds'] * 100) if trio['rounds'] > 0 else 0

            lines.append(f"{rank},{artist},{rating:.0f},{comparisons},{wins},{losses},{win_rate:.1f},{solo['rounds']},{solo['wins']},{solo_wr:.1f},{duo['rounds']},{duo['wins']},{duo_wr:.1f},{trio['rounds']},{trio['wins']},{trio_wr:.1f}")

        return "\n".join(lines)

    def format_recent_history(self, limit: int = 10) -> str:
        """Format recent comparison history for display."""
        if not self.history.records:
            return "*No comparisons yet.*"

        lines = ["*Newest first:*", ""]
        recent = self.history.records[-limit:][::-1]  # Last N, reversed (newest first)

        for i, record in enumerate(recent, 1):
            winner = record.get("winner", "?")
            artists_a = record.get("artists_a", [])
            artists_b = record.get("artists_b", [])

            winner_artists = artists_a if winner == "A" else artists_b
            loser_artists = artists_b if winner == "A" else artists_a

            winner_str = ", ".join(winner_artists)
            loser_str = ", ".join(loser_artists)

            lines.append(f"{i}. **{winner_str}** beat {loser_str}")

        return "\n".join(lines)

    def format_top_artists_display(self) -> str:
        """Format top artists for display with win rate stats."""
        top_artists = self.elo_system.get_top_artists(30)
        pool_stats = self.artist_manager.get_pool_stats()
        artist_stats = self.history.get_artist_stats()

        lines = ["## Top Artists", ""]

        if not top_artists:
            lines.append("No ratings yet. Start comparing!")
        else:
            # Use markdown list format with win rate stats
            for i, (artist, rating, comparisons) in enumerate(top_artists, 1):
                stats = artist_stats.get(artist, {})
                rounds = stats.get('rounds', 0)
                wins = stats.get('wins', 0)
                wr = (wins / rounds * 100) if rounds > 0 else 0

                # Build compact W/R breakdown
                solo = stats.get('solo', {})
                duo = stats.get('duo', {})
                trio = stats.get('trio', {})

                # Format: S:80%(5) D:70%(10) - show W/R and count for each
                wr_parts = []
                if solo.get('rounds', 0) > 0:
                    solo_wr = solo['wins'] / solo['rounds'] * 100
                    wr_parts.append(f"S:{solo_wr:.0f}%({solo['rounds']})")
                if duo.get('rounds', 0) > 0:
                    duo_wr = duo['wins'] / duo['rounds'] * 100
                    wr_parts.append(f"D:{duo_wr:.0f}%({duo['rounds']})")
                if trio.get('rounds', 0) > 0:
                    trio_wr = trio['wins'] / trio['rounds'] * 100
                    wr_parts.append(f"T:{trio_wr:.0f}%({trio['rounds']})")

                wr_breakdown = f" {' '.join(wr_parts)}" if wr_parts else ""

                lines.append(f"{i}. **{artist}** {rating:.0f} — {wr:.0f}% ({rounds})")
                if wr_breakdown.strip():
                    lines.append(f"   {wr_breakdown.strip()}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"**Comparisons:** {self.elo_system.comparison_count}  ")
        lines.append(f"**Artists rated:** {len(self.elo_system.ratings)}  ")
        lines.append(f"**Pool:** {pool_stats.get('size', 0)}/{pool_stats.get('total_artists', 0)}")

        # Pool health breakdown
        lines.append("")
        lines.append("---")
        lines.append("### Pool Health")
        safe = pool_stats.get('safe', 0)
        newcomers = pool_stats.get('newcomers', 0)
        at_risk_count = pool_stats.get('at_risk_count', 0)
        lines.append(f"Above avg: {safe}  ")
        lines.append(f"Newcomers (<5 matches): {newcomers}  ")
        lines.append(f"Below avg: {at_risk_count}")

        # Show top at-risk artists
        at_risk = pool_stats.get('at_risk', [])
        if at_risk:
            lines.append("")
            lines.append("**Most likely to rotate out:**")
            for artist, elo, matches, weight in at_risk[:5]:
                lines.append(f"- {artist} ({elo:.0f})")

        # Show recent pool changes
        if self.rotation_log:
            lines.append("")
            lines.append("---")
            lines.append("### Pool Changes")
            lines.append("*Newest first:*")
            for rot_type, artist, elo, extra in self.rotation_log[:8]:
                if rot_type == "in":
                    status = "[returning]" if extra else "[new]"
                    lines.append(f"+ **{artist}** ({elo:.0f}) {status}")
                else:
                    lines.append(f"- ~~{artist}~~ ({elo:.0f})")

        return "\n".join(lines)

    async def generate_new_comparison(self, custom_prompt: str, custom_negative_prompt: str = "", quality_toggle: bool = True, uc_preset: int = 0):
        """Generate a new pair of images for comparison."""
        # Use custom prompt if provided, otherwise default
        base_prompt = custom_prompt.strip() if custom_prompt.strip() else DEFAULT_PROMPT

        # Use custom negative prompt if provided, otherwise None (will use default)
        negative_prompt = custom_negative_prompt.strip() if custom_negative_prompt.strip() else None

        # Remove any existing artist tags from the prompt
        base_prompt = remove_artist_tags(base_prompt)

        # Ensure we have the artist placeholder or a clean prompt
        if "{artist_placeholder}" not in base_prompt:
            # Add placeholder if not present
            base_prompt = insert_artist_tags(base_prompt, "{artist_placeholder}")

        self.current_prompt = base_prompt

        try:
            session = self.get_session()
        except ValueError as e:
            # API key not configured
            error_msg = (
                "**API Key Not Configured**\n\n"
                "Please set up your NovelAI API key:\n\n"
                "1. Copy `.env.example` to `.env`\n"
                "2. Add your API key: `NOVELAI_API_KEY=pst-your-key-here`\n"
                "3. Restart the application\n\n"
                "Get your API key from [NovelAI](https://novelai.net/) → Account Settings → Get Persistent API Token"
            )
            return (
                None,
                None,
                error_msg,
                self.format_top_artists_display(),
                "",
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        path_a, path_b, artists_a, artists_b = await generate_comparison_pair(
            base_prompt,
            self.artist_manager,
            session,
            COMPARISON_IMAGES_DIR,
            negative_prompt,
            quality_toggle,
            uc_preset
        )

        if path_a and path_b:
            self.current_image_a = path_a
            self.current_image_b = path_b
            self.current_artists_a = artists_a
            self.current_artists_b = artists_b

            # Reset selection state for new comparison
            # BUT keep undo state so user can still undo the previous selection!
            self.selection_made = False
            # Don't clear last_undo_state here - it persists until next selection

            # Undo is available if we have a previous state to restore
            can_undo = self.last_undo_state is not None

            return (
                str(path_a),
                str(path_b),
                "Images generated! Pick your preferred image.",
                self.format_top_artists_display(),
                "",  # Clear result_msg
                "",  # Clear details_msg
                gr.update(interactive=True),   # Enable pick_a
                gr.update(interactive=True),   # Enable pick_b
                gr.update(interactive=can_undo),  # Undo available if we have state
            )
        else:
            return (
                None,
                None,
                "Error generating images. Please try again.",
                self.format_top_artists_display(),
                "",
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

    def pick_winner(self, winner: str):
        """Process a winner selection. Returns tuple for UI update."""
        if not self.current_artists_a or not self.current_artists_b:
            return (
                "No active comparison. Generate new images first.",
                "",
                self.format_top_artists_display(),
                gr.update(interactive=True),  # pick_a
                gr.update(interactive=True),  # pick_b
                gr.update(interactive=False),  # undo
            )

        if self.selection_made:
            return (
                "Already made a selection. Undo or generate new images.",
                "",
                self.format_top_artists_display(),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )

        if winner == "A":
            winners = self.current_artists_a
            losers = self.current_artists_b
        else:
            winners = self.current_artists_b
            losers = self.current_artists_a

        # Save state for undo BEFORE making changes
        old_ratings = {a: self.elo_system.get_rating(a) for a in winners + losers}
        old_comparisons = {a: self.elo_system.get_artist_comparison_count(a) for a in winners + losers}

        # Update ELO ratings
        self.elo_system.update_ratings(winners, losers)
        self.elo_system.save(ELO_RATINGS_FILE)

        # Update active pool (rotate losers, introduce new artists)
        rotated_out, rotated_in = self.artist_manager.process_result(winners, losers)

        # Log rotations (most recent first)
        for artist, elo, is_returning in rotated_in:
            self.rotation_log.insert(0, ("in", artist, elo, is_returning))
        for artist, elo in rotated_out:
            self.rotation_log.insert(0, ("out", artist, elo, None))
        # Keep only last 20 entries
        self.rotation_log = self.rotation_log[:20]

        # Save undo state (including current images for restoration)
        # Extract just artist names for undo
        rotated_out_names = [artist for artist, elo in rotated_out]
        self.last_undo_state = UndoState(
            winners=winners.copy(),
            losers=losers.copy(),
            old_ratings=old_ratings,
            old_comparisons=old_comparisons,
            rotated_out=rotated_out_names,
            prev_image_a=str(self.current_image_a) if self.current_image_a else None,
            prev_image_b=str(self.current_image_b) if self.current_image_b else None,
            prev_artists_a=self.current_artists_a.copy(),
            prev_artists_b=self.current_artists_b.copy(),
        )
        self.selection_made = True

        # Record history
        record = ComparisonRecord(
            timestamp=time.time(),
            artists_a=self.current_artists_a,
            artists_b=self.current_artists_b,
            winner=winner,
            image_a_path=str(self.current_image_a) if self.current_image_a else "",
            image_b_path=str(self.current_image_b) if self.current_image_b else ""
        )
        self.history.add_record(record)

        # Format result message
        winner_artists = ", ".join(winners)
        loser_artists = ", ".join(losers)
        result_msg = f"**Winner:** {winner_artists}\n**Loser:** {loser_artists}"

        # Show artist details
        details = "### Artist Details\n"
        details += f"**Image A artists:** {', '.join(self.current_artists_a)}\n"
        details += f"**Image B artists:** {', '.join(self.current_artists_b)}\n\n"
        details += "**Updated ELO ratings:**\n"
        for artist in winners + losers:
            rating = self.elo_system.get_rating(artist)
            details += f"- {artist}: {rating:.0f}\n"

        return (
            result_msg,
            details,
            self.format_top_artists_display(),
            gr.update(interactive=False),  # Disable pick_a
            gr.update(interactive=False),  # Disable pick_b
            gr.update(interactive=True),   # Enable undo
        )

    def undo_last_selection(self):
        """Undo the last selection and restore previous images."""
        if not self.last_undo_state:
            return (
                None,  # image_a
                None,  # image_b
                "Nothing to undo.",
                self.format_top_artists_display(),
                "",
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        state = self.last_undo_state

        # Restore old ratings
        for artist, old_rating in state.old_ratings.items():
            self.elo_system.ratings[artist] = old_rating

        # Restore old comparison counts
        for artist, old_count in state.old_comparisons.items():
            self.elo_system.artist_comparisons[artist] = old_count

        # Decrement total comparison count
        self.elo_system.comparison_count -= 1
        self.elo_system.save(ELO_RATINGS_FILE)

        # Restore rotated out artists to pool
        if state.rotated_out:
            self.artist_manager.restore_artists(state.rotated_out)

        # Remove last history record
        if self.history.records:
            self.history.records.pop()
            self.history.save()

        # Restore previous images and artists
        self.current_image_a = Path(state.prev_image_a) if state.prev_image_a else None
        self.current_image_b = Path(state.prev_image_b) if state.prev_image_b else None
        self.current_artists_a = state.prev_artists_a.copy()
        self.current_artists_b = state.prev_artists_b.copy()

        # Clear undo state and reset selection
        self.last_undo_state = None
        self.selection_made = False

        return (
            state.prev_image_a,  # image_a
            state.prev_image_b,  # image_b
            "**Undone!** Pick again.",
            self.format_top_artists_display(),
            "",  # Clear result_msg
            "",  # Clear details_msg
            gr.update(interactive=True),   # Enable pick_a
            gr.update(interactive=True),   # Enable pick_b
            gr.update(interactive=False),  # Disable undo
        )

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface."""

        with gr.Blocks(title="Artist ELO Ranker") as app:
            gr.Markdown("# Artist ELO Ranking System")
            gr.Markdown(
                "Compare images generated with different artist combinations. "
                "Pick your preferred image to update ELO ratings. "
                "The artist tags are hidden during comparison for unbiased selection.  \n"
                "**Shortcuts:** `1` = Pick A, `2` = Pick B, `s` = Skip, `0` = Undo"
            )

            with gr.Row():
                # Main comparison area (left side)
                with gr.Column(scale=2):
                    # Prompt input
                    with gr.Accordion("Custom Prompts (Optional)", open=False):
                        prompt_input = gr.Textbox(
                            label="Positive Prompt",
                            placeholder="Enter custom prompt (artist tags will be auto-inserted)...",
                            lines=4,
                            value=""
                        )
                        negative_prompt_input = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Enter custom negative prompt (leave empty for default)...",
                            lines=3,
                            value=""
                        )
                        with gr.Row():
                            quality_toggle = gr.Checkbox(
                                label="Add quality tags",
                                value=True,
                                info="Adds 'very aesthetic, masterpiece, no text' to prompt"
                            )
                            uc_preset_dropdown = gr.Dropdown(
                                label="Auto-Negative Preset",
                                choices=[
                                    ("None - no auto-negatives", -1),
                                    ("Heavy - standard quality filters", 0),
                                    ("Light - minimal quality filters", 1),
                                    ("Human Focus - optimized for characters", 2),
                                    ("Heavy + Anatomy - includes body fixes", 3),
                                ],
                                value=0,
                                interactive=True,
                            )
                        gr.Markdown(
                            "*Leave prompts empty to use defaults. Artist tags are only inserted into the positive prompt.*"
                        )

                    # Status message
                    status_msg = gr.Markdown("Generating first comparison...")

                    # Image comparison
                    with gr.Row():
                        with gr.Column():
                            image_a = gr.Image(label="Image A", type="filepath")
                            artists_a_display = gr.Markdown("", visible=False)
                            pick_a_btn = gr.Button("Pick Image A", variant="secondary", size="lg")

                        with gr.Column():
                            image_b = gr.Image(label="Image B", type="filepath")
                            artists_b_display = gr.Markdown("", visible=False)
                            pick_b_btn = gr.Button("Pick Image B", variant="secondary", size="lg")

                    # Show artists toggle, skip, and undo buttons
                    with gr.Row():
                        show_artists_toggle = gr.Checkbox(label="Show artist tags", value=False)
                        skip_btn = gr.Button("Skip (both bad)", variant="secondary", size="sm")
                        undo_btn = gr.Button("Undo Last Selection", variant="stop", size="sm", interactive=False)

                    # Result display
                    result_msg = gr.Markdown("")
                    details_msg = gr.Markdown("")

                # Leaderboard (right side)
                with gr.Column(scale=1):
                    leaderboard = gr.Markdown(
                        self.format_top_artists_display(),
                        label="Top Artists"
                    )
                    export_btn = gr.Button("Export Leaderboard as CSV")
                    export_file = gr.File(label="Download", visible=False)

                    # Comparison history panel
                    with gr.Accordion("Comparison History", open=False):
                        history_display = gr.Markdown(self.format_recent_history())

            # Event handlers
            def on_generate(prompt, negative_prompt, quality_tags, uc_preset):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.generate_new_comparison(prompt, negative_prompt, quality_tags, uc_preset))
                    # Add artist display text and history to result
                    artists_a_text = f"**Artists:** {', '.join(self.current_artists_a)}"
                    artists_b_text = f"**Artists:** {', '.join(self.current_artists_b)}"
                    history_text = self.format_recent_history()
                    return result + (artists_a_text, artists_b_text, history_text)
                finally:
                    loop.close()

            def on_pick_then_generate(prompt, negative_prompt, quality_tags, uc_preset):
                """Generate new comparison after pick (for chaining)."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.generate_new_comparison(prompt, negative_prompt, quality_tags, uc_preset))
                    artists_a_text = f"**Artists:** {', '.join(self.current_artists_a)}"
                    artists_b_text = f"**Artists:** {', '.join(self.current_artists_b)}"
                    history_text = self.format_recent_history()
                    return result + (artists_a_text, artists_b_text, history_text)
                finally:
                    loop.close()

            def on_export():
                """Export leaderboard as downloadable CSV file."""
                content = self.export_leaderboard_csv()
                filepath = COMPARISON_IMAGES_DIR / "leaderboard_export.csv"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                return str(filepath)

            def on_toggle_artists(show):
                """Toggle visibility of artist tags."""
                return (
                    gr.update(visible=show),
                    gr.update(visible=show),
                )

            def on_undo():
                """Undo and restore artist display text."""
                result = self.undo_last_selection()
                artists_a_text = f"**Artists:** {', '.join(self.current_artists_a)}" if self.current_artists_a else ""
                artists_b_text = f"**Artists:** {', '.join(self.current_artists_b)}" if self.current_artists_b else ""
                history_text = self.format_recent_history()
                return result + (artists_a_text, artists_b_text, history_text)

            # Auto-generate first comparison on app load
            app.load(
                fn=on_generate,
                inputs=[prompt_input, negative_prompt_input, quality_toggle, uc_preset_dropdown],
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            )

            # Pick A: update ELO, then auto-generate next pair
            pick_a_btn.click(
                fn=lambda: self.pick_winner("A"),
                outputs=[result_msg, details_msg, leaderboard, pick_a_btn, pick_b_btn, undo_btn]
            ).then(
                fn=on_pick_then_generate,
                inputs=[prompt_input, negative_prompt_input, quality_toggle, uc_preset_dropdown],
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            )

            # Pick B: update ELO, then auto-generate next pair
            pick_b_btn.click(
                fn=lambda: self.pick_winner("B"),
                outputs=[result_msg, details_msg, leaderboard, pick_a_btn, pick_b_btn, undo_btn]
            ).then(
                fn=on_pick_then_generate,
                inputs=[prompt_input, negative_prompt_input, quality_toggle, uc_preset_dropdown],
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            )

            # Undo: restore previous state and images
            undo_btn.click(
                fn=on_undo,
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            )

            # Toggle artist visibility
            show_artists_toggle.change(
                fn=on_toggle_artists,
                inputs=[show_artists_toggle],
                outputs=[artists_a_display, artists_b_display]
            )

            # Skip: generate new images without any ELO changes
            skip_btn.click(
                fn=on_pick_then_generate,
                inputs=[prompt_input, negative_prompt_input, quality_toggle, uc_preset_dropdown],
                outputs=[image_a, image_b, status_msg, leaderboard, result_msg, details_msg, pick_a_btn, pick_b_btn, undo_btn, artists_a_display, artists_b_display, history_display]
            )

            # Export leaderboard - generate CSV and show download link
            export_btn.click(
                fn=on_export,
                outputs=[export_file]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[export_file]
            )

            # Keyboard shortcuts via JavaScript
            app.load(
                fn=None,
                js="""
                () => {
                    document.addEventListener('keydown', (e) => {
                        // Ignore if typing in a text field
                        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                        const findAndClick = (text) => {
                            const btns = document.querySelectorAll('button');
                            for (const b of btns) {
                                if (b.textContent.includes(text)) {
                                    // Check if not disabled (Gradio uses multiple ways)
                                    if (!b.disabled && !b.classList.contains('disabled') && b.getAttribute('aria-disabled') !== 'true') {
                                        b.click();
                                        return true;
                                    }
                                }
                            }
                            return false;
                        };

                        if (e.key === '1') {
                            findAndClick('Pick Image A');
                        } else if (e.key === '2') {
                            findAndClick('Pick Image B');
                        } else if (e.key === 's' || e.key === 'S') {
                            findAndClick('Skip');
                        } else if (e.key === '0') {
                            findAndClick('Undo');
                        }
                    });
                    return [];
                }
                """
            )

        return app


# --------------------------------------------------------------------------------
# Main Entry Point
# --------------------------------------------------------------------------------

def main():
    """Main entry point."""
    print("Starting Artist ELO Ranker...")
    print(f"Artist tags file: {ARTIST_TAGS_FILE}")
    print(f"ELO ratings file: {ELO_RATINGS_FILE}")
    print(f"Comparison images dir: {COMPARISON_IMAGES_DIR}")

    ranker = ArtistELORanker()
    app = ranker.create_ui()

    # Launch the app
    app.launch(
        share=False,
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        inbrowser=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
