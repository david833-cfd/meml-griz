# meml_griz
Memphis Grizzlies ML (`meml_griz`)

This project uses the [nba_api](https://github.com/swar/nba_api) python project to download NBA game data and create ML models.

## Goals
### 1. Create a data pipeline that gathers NBA stats.
- `src/meml_griz/fetch.py`: Fetch NBA data from `nba_api`
- `sandbox/fill_cache.ipynb`: Download game data to local machine.

### 2. Create data classes that allow comparison of scikit-learn and pytorch models.
- `src/meml_griz/snifcheck.py`

### 3. Try different ML methods on the game datasets.
- `sandbox/win_or_lose/`
