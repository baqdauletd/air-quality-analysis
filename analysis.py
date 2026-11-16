"""Exploratory data analysis and visualizations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config
from utils import ensure_dirs, has_module

sns.set_theme(style="whitegrid")


def describe_data(df):
    """Descriptive statistics of all numeric columns."""
    return df[config.NUMERIC_COLS].describe().T.round(2)


def iqr_outliers(df, cols=None):
    """Number and share of outliers per column using the 1.5*IQR rule."""
    rows = []
    for col in cols or config.POLLUTANTS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"column": col, "lower_fence": round(low, 1), "upper_fence": round(high, 1),
                     "outliers": n, "percent": round(100 * n / df[col].notna().sum(), 2)})
    return pd.DataFrame(rows)


def correlation_matrix(df, cols=None):
    return df[cols or config.NUMERIC_COLS].corr().round(2)


def _save(fig, name):
    ensure_dirs()
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_boxplot_by_station(df):
    order = df.groupby("station", observed=True)["PM2.5"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="station", y="PM2.5", order=order, showfliers=False, ax=ax, color="#8fb8de")
    ax.set_title("PM2.5 distribution by station (outliers hidden, IQR boxes)")
    ax.set_ylabel("PM2.5, ug/m3")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, "01_boxplot_pm25_by_station.png")


def plot_monthly_by_area(df):
    monthly = (df.set_index("ts").groupby("area_type", observed=True)["PM2.5"]
               .resample("MS").mean().reset_index())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.lineplot(data=monthly, x="ts", y="PM2.5", hue="area_type", marker="o", ax=ax)
    ax.set_title("Monthly mean PM2.5: urban vs suburban vs rural stations")
    ax.set_xlabel("")
    ax.set_ylabel("PM2.5, ug/m3")
    return _save(fig, "02_monthly_pm25_by_area_type.png")


def plot_season_by_station(df):
    table = (df.groupby(["station", "season"], observed=True)["PM2.5"].mean().reset_index())
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=table, x="station", y="PM2.5", hue="season", hue_order=config.SEASON_ORDER, ax=ax)
    ax.set_title("Seasonal mean PM2.5 by station")
    ax.set_ylabel("PM2.5, ug/m3")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, "03_season_pm25_by_station.png")


def plot_hourly_profile(df):
    table = df.groupby(["hour", "season"], observed=True)["PM2.5"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(data=table, x="hour", y="PM2.5", hue="season", hue_order=config.SEASON_ORDER, ax=ax)
    ax.set_title("Daily cycle of PM2.5 by season")
    ax.set_xlabel("Hour of day")
    return _save(fig, "04_hourly_profile_by_season.png")


def plot_correlation(df):
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation: pollutants and weather")
    return _save(fig, "05_correlation_heatmap.png")


def plot_interactive(df):
    """Interactive Plotly chart (HTML). Returns None if plotly is not installed."""
    if not has_module("plotly"):
        return None
    import plotly.express as px
    daily = (df.set_index("ts").groupby("station", observed=True)["PM2.5"]
             .resample("D").mean().reset_index())
    fig = px.line(daily, x="ts", y="PM2.5", color="station",
                  title="Daily mean PM2.5 by station (interactive: zoom, hover, click legend)")
    fig.update_xaxes(rangeslider_visible=True)
    path = config.FIG_DIR / "06_interactive_daily_pm25.html"
    ensure_dirs()
    fig.write_html(path)
    return path


def run_eda(df):
    """Run the whole EDA step; saves tables and figures."""
    ensure_dirs()
    describe_data(df).to_csv(config.TABLE_DIR / "descriptive_stats.csv")
    iqr_outliers(df).to_csv(config.TABLE_DIR / "iqr_outliers.csv", index=False)
    correlation_matrix(df).to_csv(config.TABLE_DIR / "correlation.csv")
    figures = [plot_boxplot_by_station(df), plot_monthly_by_area(df), plot_season_by_station(df),
               plot_hourly_profile(df), plot_correlation(df), plot_interactive(df)]
    return [f for f in figures if f is not None]
