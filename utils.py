"""Small helper functions shared by all modules."""
import json
import time
import tracemalloc

import config


def ensure_dirs():
    """Create all output folders if they do not exist."""
    for folder in (config.PROCESSED_DIR, config.FIG_DIR, config.TABLE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def measure(func, *args, **kwargs):
    """Run func and return (result, seconds, peak_memory_MB)."""
    tracemalloc.start()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak / 1024 ** 2


def ask_choice(prompt, valid):
    """Keep asking until the user types one of the valid options."""
    while True:
        value = input(prompt).strip()
        if value in valid:
            return value
        print(f"  Please choose one of: {', '.join(valid)}")


def has_module(name):
    """True if an optional package (plotly, statsmodels, pyarrow...) is installed."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False
