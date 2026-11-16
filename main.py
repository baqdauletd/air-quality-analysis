"""Console menu for the Beijing air-quality project.

Run:  python main.py          (interactive menu)
      python main.py --all    (run every step without the menu)
"""
import argparse

import pandas as pd

import analysis
import benchmarks
import config
import data_loader
import db
import forecast
import timeseries
from utils import ask_choice, ensure_dirs

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

MENU = """
==== Beijing Air Quality - menu ====
 1 - Load and clean the data
 2 - Build the SQLite database (+ ER diagram)
 3 - Run the 10 SQL queries
 4 - EDA and figures
 5 - Time series: decomposition and moving averages
 6 - Forecast: baselines vs Linear Regression
 7 - Compare processing approaches (pandas / SQL / Parquet)
 8 - Look up PM2.5 for a station and date
 9 - Run everything
 0 - Exit
"""


def step_load():
    df, summary = data_loader.run_pipeline()
    print("Loaded and cleaned:", summary)
    return df


def step_database(df):
    counts = db.build_database(df)
    print("Database built:", counts)
    print("ER diagram saved to", db.draw_er_diagram())


def step_queries():
    for name, frame in db.run_all_queries().items():
        print(f"\n--- {name}: {db.QUERIES[name][0]}")
        print(frame.head(8).to_string(index=False))


def step_eda(df):
    print(analysis.iqr_outliers(df).to_string(index=False))
    print(analysis.correlation_matrix(df)["PM2.5"].to_string())
    for path in analysis.run_eda(df):
        print("saved", path.name)


def step_timeseries(df):
    result = timeseries.run_timeseries(df)
    print(f"Decomposition method: {result['method']}; seasonal strength = {result['strength']:.2f}")
    print(result["seasonal_profile"].to_string())
    print(result["table"].groupby(["area_type", "pollutant"])[
        ["seasonal_amplitude", "seasonal_strength"]].mean().round(2))


def step_forecast(df):
    table, figure = forecast.run_forecast(df)
    print(table.to_string(index=False))
    print("figure:", figure.name)


def step_benchmarks(df):
    table, sizes = benchmarks.run_benchmarks(df)
    print(table.to_string(index=False))
    print(sizes.to_string(index=False))


def step_lookup(df):
    stations = sorted(df["station"].astype(str).unique())
    print("Stations:", ", ".join(stations))
    name = input("Station name: ").strip().title()
    if name not in stations:
        print("Unknown station.")
        return
    try:
        day = pd.Timestamp(input("Date (YYYY-MM-DD): ").strip())
    except ValueError:
        print("Invalid date format.")
        return
    part = df[(df["station"].astype(str) == name) & (df["ts"].dt.normalize() == day)]
    if part.empty:
        print("No data for this date (data covers 2013-03-01 .. 2017-02-28).")
    else:
        print(part[["ts", "PM2.5", "PM10", "NO2", "TEMP", "WSPM"]].to_string(index=False))
        print(f"Daily mean PM2.5: {part['PM2.5'].mean():.1f} ug/m3")


def run_all():
    df = step_load()
    step_database(df)
    step_queries()
    step_eda(df)
    step_timeseries(df)
    step_forecast(df)
    step_benchmarks(df)
    return df


def menu():
    df = None
    actions = {"1": "load", "2": "db", "3": "queries", "4": "eda", "5": "ts", "6": "fc",
               "7": "bench", "8": "lookup", "9": "all", "0": "exit"}
    while True:
        print(MENU)
        choice = ask_choice("Choose an option: ", list(actions))
        action = actions[choice]
        if action == "exit":
            print("Bye!")
            break
        try:
            if action == "load":
                df = step_load()
            elif action == "all":
                df = run_all()
            elif action == "queries":
                step_queries()
            else:
                if df is None:
                    df = data_loader.load_clean()      # raises FileNotFoundError if step 1 was never run
                {"db": lambda: step_database(df), "eda": lambda: step_eda(df),
                 "ts": lambda: step_timeseries(df), "fc": lambda: step_forecast(df),
                 "bench": lambda: step_benchmarks(df), "lookup": lambda: step_lookup(df)}[action]()
        except FileNotFoundError as err:
            print("Error:", err)


if __name__ == "__main__":
    ensure_dirs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run every step and exit")
    args = parser.parse_args()
    run_all() if args.all else menu()
