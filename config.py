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
PROFILES_DIR = SCRIPT_DIR / "profiles"
COMPARISON_IMAGES_DIR = SCRIPT_DIR / "comparison_images"

# Default profile paths (used when no profile system, for backwards compatibility)
ELO_RATINGS_FILE = SCRIPT_DIR / "artist_elo_ratings.json"
COMPARISON_HISTORY_FILE = SCRIPT_DIR / "comparison_history.json"
ACTIVE_POOL_FILE = SCRIPT_DIR / "active_pool.json"


def get_profile_dir(profile_name: str) -> Path:
    """Get the directory for a specific profile."""
    return PROFILES_DIR / profile_name


def get_profile_files(profile_name: str) -> dict:
    """Get all file paths for a profile."""
    profile_dir = get_profile_dir(profile_name)
    return {
        "elo_ratings": profile_dir / "artist_elo_ratings.json",
        "active_pool": profile_dir / "active_pool.json",
        "comparison_history": profile_dir / "comparison_history.json",
        "settings": profile_dir / "settings.json",
    }


def list_profiles() -> list:
    """List all available profiles."""
    if not PROFILES_DIR.exists():
        return []
    return sorted([d.name for d in PROFILES_DIR.iterdir() if d.is_dir()])


def create_profile(profile_name: str) -> bool:
    """Create a new profile directory."""
    profile_dir = get_profile_dir(profile_name)
    if profile_dir.exists():
        return False
    profile_dir.mkdir(parents=True, exist_ok=True)
    return True


def delete_profile(profile_name: str) -> bool:
    """Delete a profile directory and all its contents."""
    import shutil
    profile_dir = get_profile_dir(profile_name)
    if not profile_dir.exists():
        return False
    shutil.rmtree(profile_dir)
    return True


def migrate_legacy_data_to_profile(profile_name: str) -> bool:
    """
    Migrate legacy root-level data files to a profile.
    Returns True if migration occurred, False if no legacy data existed.
    """
    import shutil
    profile_files = get_profile_files(profile_name)
    profile_dir = profile_files["elo_ratings"].parent
    profile_dir.mkdir(parents=True, exist_ok=True)

    migrated = False

    # Migrate ELO ratings
    if ELO_RATINGS_FILE.exists() and not profile_files["elo_ratings"].exists():
        shutil.copy(ELO_RATINGS_FILE, profile_files["elo_ratings"])
        migrated = True

    # Migrate active pool
    if ACTIVE_POOL_FILE.exists() and not profile_files["active_pool"].exists():
        shutil.copy(ACTIVE_POOL_FILE, profile_files["active_pool"])
        migrated = True

    # Migrate comparison history
    if COMPARISON_HISTORY_FILE.exists() and not profile_files["comparison_history"].exists():
        shutil.copy(COMPARISON_HISTORY_FILE, profile_files["comparison_history"])
        migrated = True

    return migrated


def ensure_default_profile() -> str:
    """
    Ensure a default profile exists. If legacy data exists, migrate it.
    Returns the name of the default profile.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    profiles = list_profiles()

    # If profiles exist, return the first one
    if profiles:
        return profiles[0]

    # Create default profile and migrate any existing data
    default_name = "default"
    create_profile(default_name)
    migrate_legacy_data_to_profile(default_name)

    return default_name


# --------------------------------------------------------------------------------
# NovelAI Generation Parameters
# --------------------------------------------------------------------------------

# These can be overridden via environment variables if needed
STEPS = int(os.getenv("NAI_STEPS", "28"))
IMG_WIDTH = int(os.getenv("NAI_IMG_WIDTH", "1024"))
IMG_HEIGHT = int(os.getenv("NAI_IMG_HEIGHT", "1024"))


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
