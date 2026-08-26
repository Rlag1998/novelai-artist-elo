"""
Configuration management for Artist ELO Ranker.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# --------------------------------------------------------------------------------
# API Configuration
# --------------------------------------------------------------------------------

def get_api_key() -> str:
    """
    Get NovelAI API key from environment variable.

    Raises:
        ValueError: If NOVELAI_API_KEY is not set
    """
    api_key = os.getenv("NOVELAI_API_KEY")
    if not api_key:
        raise ValueError(
            "NOVELAI_API_KEY environment variable is not set.\n"
            "Please set it in your .env file or export it:\n"
            "  export NOVELAI_API_KEY='your-api-key-here'\n\n"
            "You can get an API key from https://novelai.net/ (requires subscription)"
        )
    return api_key


# --------------------------------------------------------------------------------
# File Paths
# --------------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
ARTIST_TAGS_FILE = SCRIPT_DIR / "danbooru_artist_tags_v4.5.txt"
COMPARISON_IMAGES_DIR = SCRIPT_DIR / "comparison_images"
ELO_RATINGS_FILE = SCRIPT_DIR / "artist_elo_ratings.json"
COMPARISON_HISTORY_FILE = SCRIPT_DIR / "comparison_history.json"
ACTIVE_POOL_FILE = SCRIPT_DIR / "active_pool.json"
SKILL_FILE = SCRIPT_DIR / "artist_skill.json"


# --------------------------------------------------------------------------------
# NovelAI Generation Parameters
# --------------------------------------------------------------------------------

# These can be overridden via environment variables if needed
STEPS = int(os.getenv("NAI_STEPS", "28"))
IMG_WIDTH = int(os.getenv("NAI_IMG_WIDTH", "1024"))
IMG_HEIGHT = int(os.getenv("NAI_IMG_HEIGHT", "1024"))

# NovelAI model id. Defaults to NovelAI Diffusion V5 Full. Set NAI_MODEL to pin
# another model, for example nai-diffusion-4-5-full.
MODEL_ID = os.getenv("NAI_MODEL", "nai-diffusion-5-full")

# Seed handling for the two images of a round. "shared" gives both images the
# same seed so only the artist tags differ, which removes composition luck from
# the comparison. "independent" restores the old behaviour of a fresh seed each.
SEED_MODE = os.getenv("NAI_SEED_MODE", "shared").strip().lower()


# --------------------------------------------------------------------------------
# ELO System Parameters
# --------------------------------------------------------------------------------

DEFAULT_ELO = int(os.getenv("ELO_DEFAULT", "1500"))
K_FACTOR = int(os.getenv("ELO_K_FACTOR", "32"))


# --------------------------------------------------------------------------------
# Active Pool Settings
# --------------------------------------------------------------------------------

ACTIVE_POOL_SIZE = int(os.getenv("POOL_SIZE", "150"))
NEW_ARTIST_PROBABILITY = float(os.getenv("NEW_ARTIST_PROB", "0.15"))
LOSER_ROTATION_PROBABILITY = float(os.getenv("LOSER_ROTATION_PROB", "0.4"))


# --------------------------------------------------------------------------------
# Skill Model and Pair Selection
# --------------------------------------------------------------------------------

# An artist counts as "settled" once its TrueSkill uncertainty (sigma) is at or
# below this. Fresh artists start at 8.33.
SETTLED_SIGMA = float(os.getenv("SETTLED_SIGMA", "3.0"))

# "random" (default): the pool's own weighting, which favours under-compared
# artists. "skill" is experimental: side B is chosen from sampled candidates by
# TrueSkill match quality. In offline simulation (scripts/simulate_pairing.py)
# "skill" recovered the true ranking worse than "random" at every judge-noise
# level tested, so it is off by default.
MATCHMAKING = os.getenv("MATCHMAKING", "random").strip().lower()
MATCH_CANDIDATES = int(os.getenv("MATCH_CANDIDATES", "30"))
# Share of rounds that stay fully random so the matchmaker cannot lock in.
EXPLORE_RATE = float(os.getenv("EXPLORE_RATE", "0.2"))


# --------------------------------------------------------------------------------
# Server Settings
# --------------------------------------------------------------------------------

SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))


# --------------------------------------------------------------------------------
# Default Prompts
# --------------------------------------------------------------------------------

NEGATIVE_PROMPT = os.getenv("NEGATIVE_PROMPT", (
    "elf ears, animal ears, horns, pig nose, furry, pencil sketch, {{{multiple angles}}}, {{2 people}}, "
    "{{{{concept art}}}}, {{multiple faces}}, {{{{character sheet}}}}, {{concept art}}, speech bubble, "
    "{{multiple characters}}, {chibi}, {lolicon}, caricature, photo frame, circular frame, circular border, "
    "black border, watermark, {{stretched earlobes}}, {{{text}}}, {caption}, numbers, "
    "mutated earlobes, worst quality, lowres, jpeg artefacts, blurry, ugly, gross proportions, "
    "{{{large earlobes}}}, worst anatomy, mutated fingers, bad hands, big ears, bad feet, extra digit, "
))

DEFAULT_PROMPT = os.getenv("DEFAULT_PROMPT", (
    "1girl, a Neo-Solar Hegemony woman, Toiling as a Hydroponic Engineer, refining sunburst harvests, "
    "synchronizing cycles, enhancing prestige, Dressed in exo-fabric bodysuits, bright gold trim, "
    "prestige detailing, {artist_placeholder}, location, very aesthetic, masterpiece, very tall height, "
    "mesomorphic, slightly toned, brown skin, aurora strawberry hair, straight hair, thick hair, "
    "classic oval shaped face, cosmic honey eyes, round eyes, wide set eyes, protruding eyes, "
    "aurora strawberry eyebrows, medium thickness eyebrows, s-shaped eyebrows, medium length eyelashes, "
    "greek nose, tall lips, detached earlobes, mole on face, dimples, a reserved but intense demeanor,"
))
