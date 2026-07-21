from pathlib import Path

CACHE_DIR = Path().home() / ".meml_griz" / "cache"
DB_DIR = CACHE_DIR.parent.resolve()

CACHE_DIR.mkdir(exist_ok=True, parents=True)
DB_DIR.mkdir(exist_ok=True, parents=True)
SLEEP_MIN, SLEEP_MAX = 0.1, 3.0
