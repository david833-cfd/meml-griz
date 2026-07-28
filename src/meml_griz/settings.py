from pathlib import Path
from nba_api.stats.library.http import NBAStatsHTTP

CACHE_DIR = Path().home() / ".meml_griz" / "cache"
DB_DIR = CACHE_DIR.parent.resolve()

CACHE_DIR.mkdir(exist_ok=True, parents=True)
DB_DIR.mkdir(exist_ok=True, parents=True)
SLEEP_MIN, SLEEP_MAX = 0.1, 10.0

# Force nba_api to act as a normal web browser
custom_headers = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://nba.com",
    "Connection": "keep-alive",
}

NBAStatsHTTP.default_headers = custom_headers
