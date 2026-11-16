"""Loading, cleaning and storing the Beijing air-quality data."""
import pandas as pd

import config
from utils import ensure_dirs, has_module, save_json

DTYPES = {
    "No": "int32", "year": "int16", "month": "int8", "day": "int8", "hour": "int8",
    "PM2.5": "float32", "PM10": "float32", "SO2": "float32", "NO2": "float32",
    "CO": "float32", "O3": "float32", "TEMP": "float32", "PRES": "float32",
    "DEWP": "float32", "RAIN": "float32", "WSPM": "float32",
    "wd": "category", "station": "category",
}


def list_station_files(raw_dir=config.RAW_DIR):
    files = sorted(raw_dir.glob("PRSA_Data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No PRSA_Data_*.csv files found in {raw_dir}")
    return files


def read_station_csv(path, chunksize=20000):
    """Read one station file in chunks with compact dtypes."""
    parts = []
    for chunk in pd.read_csv(path, dtype=DTYPES, na_values=["NA"], chunksize=chunksize):
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def load_raw(raw_dir=config.RAW_DIR):
    """Load all 12 stations into one DataFrame."""
    frames = [read_station_csv(p) for p in list_station_files(raw_dir)]
    df = pd.concat(frames, ignore_index=True)
    df["station"] = df["station"].astype("category")
    df["wd"] = df["wd"].astype("category")
    return df


def memory_mb(df):
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def load_raw_unoptimized(raw_dir=config.RAW_DIR):
    """Default pandas dtypes - used only to measure the memory saving."""
    return pd.concat([pd.read_csv(p) for p in list_station_files(raw_dir)], ignore_index=True)


def missing_report(df):
    """Missing values per column (count and percent)."""
    cols = config.NUMERIC_COLS + ["wd"]
    miss = df[cols].isna().sum()
    return pd.DataFrame({"missing": miss, "percent": (miss / len(df) * 100).round(2)})


def fill_short_gaps(series, max_gap, method="interpolate"):
    """Fill runs of missing values that are at most `max_gap` long; leave longer runs untouched.

    (pandas `limit=` alone would fill the first `max_gap` values of a longer gap.)
    """
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    if method == "ffill":
        filled = series.ffill()
    else:
        filled = series.interpolate(limit_area="inside")
    return filled.where(~(missing & (run_length > max_gap)))


def clean_data(df, max_gap=config.MAX_GAP_HOURS):
    """Build timestamp, remove duplicates, fill short gaps, add area type."""
    df = df.drop(columns=["No"], errors="ignore").copy()
    df["ts"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    before = len(df)
    df = df.drop_duplicates(subset=["station", "ts"])
    removed = before - len(df)
    df = df.sort_values(["station", "ts"]).reset_index(drop=True)

    # fill gaps of at most `max_gap` hours inside each station (hourly data);
    # longer gaps are left completely missing
    grouped = df.groupby("station", observed=True)
    for col in config.NUMERIC_COLS:
        df[col] = grouped[col].transform(lambda s: fill_short_gaps(s, max_gap))
    df["wd"] = grouped["wd"].transform(lambda s: fill_short_gaps(s.astype(object), max_gap, method="ffill"))
    df["wd"] = df["wd"].astype("category")
    df["area_type"] = df["station"].astype(str).map(config.AREA_TYPE).astype("category")
    df["season"] = df["month"].map(config.SEASONS).astype("category")
    return df, removed


def run_pipeline():
    """Full loading + cleaning step. Returns the clean DataFrame and a summary dict."""
    ensure_dirs()
    raw = load_raw()
    raw_missing = missing_report(raw)
    raw_mem = memory_mb(raw)
    default_mem = memory_mb(load_raw_unoptimized())
    clean, removed = clean_data(raw)
    clean_missing = missing_report(clean)

    raw_missing.join(clean_missing, rsuffix="_after", lsuffix="_before").to_csv(
        config.TABLE_DIR / "missing_values.csv")
    summary = {
        "rows": len(clean), "stations": int(clean["station"].nunique()),
        "period": [str(clean["ts"].min()), str(clean["ts"].max())],
        "duplicates_removed": int(removed),
        "memory_default_dtypes_mb": round(float(default_mem), 1),
        "memory_optimized_dtypes_mb": round(float(raw_mem), 1),
    }
    save_json(summary, config.TABLE_DIR / "load_summary.json")
    save_clean(clean)
    return clean, summary


def save_clean(df):
    """Persist cleaned data: CSV always, Parquet if pyarrow is installed."""
    ensure_dirs()
    df.to_csv(config.CLEAN_CSV, index=False)
    if has_module("pyarrow"):
        df.to_parquet(config.CLEAN_PARQUET, index=False)


def load_clean():
    """Load the cleaned data (Parquet preferred, otherwise CSV)."""
    try:
        if config.CLEAN_PARQUET.exists() and has_module("pyarrow"):
            return pd.read_parquet(config.CLEAN_PARQUET)
        df = pd.read_csv(config.CLEAN_CSV, parse_dates=["ts"], dtype={
            "wd": "category", "station": "category", "area_type": "category", "season": "category"})
        return df
    except FileNotFoundError:
        raise FileNotFoundError("Clean data not found. Run option 1 (load & clean) first.")
