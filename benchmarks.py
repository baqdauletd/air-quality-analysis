import time

import pandas as pd

import config
import db
import data_loader
from utils import has_module, measure


def _pandas_monthly(df):
    ts = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.groupby(["station", ts], observed=True)["PM2.5"].mean()


def _sql_monthly():
    return db.run_query("""SELECT s.name, strftime('%Y-%m', p.ts) AS ym, AVG(p.pm25) AS pm25
                           FROM pollution p JOIN stations s USING (station_id)
                           GROUP BY s.name, ym""")


def run_benchmarks(clean_df):
    rows = []

    def add(name, func):
        result, seconds, peak = measure(func)
        rows.append({"approach": name, "time_s": round(seconds, 3), "peak_memory_MB": round(peak, 1)})
        return result

    def baseline():
        raw = data_loader.load_raw_unoptimized()
        return _pandas_monthly(raw)

    add("Baseline: pandas, raw CSV, default dtypes (load + aggregate)", baseline)
    add("A: pandas, optimised dtypes, cleaned CSV (load + aggregate)",
        lambda: _pandas_monthly(pd.read_csv(config.CLEAN_CSV,
                                            usecols=["station", "year", "month", "PM2.5"],
                                            dtype={"station": "category"})))
    add("B: SQL (SQLite GROUP BY, indexed)", _sql_monthly)
    if has_module("pyarrow") and config.CLEAN_PARQUET.exists():
        add("C: Parquet (column read + aggregate)",
            lambda: _pandas_monthly(pd.read_parquet(config.CLEAN_PARQUET,
                                                    columns=["station", "year", "month", "PM2.5"])))
    else:
        rows.append({"approach": "C: Parquet (skipped - pyarrow not installed)", "time_s": None,
                     "peak_memory_MB": None})
    table = pd.DataFrame(rows)

    sizes = {"raw CSV files (12)": sum(p.stat().st_size for p in config.RAW_DIR.glob("*.csv")),
             "clean CSV": config.CLEAN_CSV.stat().st_size if config.CLEAN_CSV.exists() else None,
             "SQLite database": config.DB_PATH.stat().st_size if config.DB_PATH.exists() else None}
    if config.CLEAN_PARQUET.exists():
        sizes["clean Parquet"] = config.CLEAN_PARQUET.stat().st_size
    size_table = pd.DataFrame({"storage": list(sizes), "size_MB": [round(v / 1024 ** 2, 1) if v else None
                                                                   for v in sizes.values()]})
    table.to_csv(config.TABLE_DIR / "benchmark_processing.csv", index=False)
    size_table.to_csv(config.TABLE_DIR / "benchmark_storage_sizes.csv", index=False)
    return table, size_table
