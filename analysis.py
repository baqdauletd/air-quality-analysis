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


def plot_mean_vs_median(df, col="PM2.5", filename="10_mean_vs_median_pm25.png"):
    """Dumbbell chart: median and mean of hourly values per station (the gap shows the skew)."""
    stats = df.groupby("station", observed=True)[col].agg(["mean", "median"]).sort_values("mean")
    rows = list(range(len(stats)))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hlines(rows, stats["median"], stats["mean"], color="#9aa5ad", linewidth=2, zorder=1)
    ax.scatter(stats["median"], rows, color="#E07A2F", s=40, label="median", zorder=2)
    ax.scatter(stats["mean"], rows, color="#0E7C86", s=40, label="mean", zorder=2)
    ax.set_yticks(rows)
    ax.set_yticklabels(stats.index, fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel(f"{col}, ug/m3 (hourly values)", fontsize=10)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.grid(axis="y", visible=False)
    return _save(fig, filename)


def plot_monthly_mean(df, col="PM2.5", filename="11_monthly_mean_pm25.png"):
    """Average value for each calendar month (all stations, all years); winter and summer highlighted."""
    monthly = df.groupby(df["ts"].dt.month)[col].mean()
    colors = ["#0E7C86" if m in (12, 1, 2) else "#E07A2F" if m in (6, 7, 8) else "#B7C4CB" for m in monthly.index]
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(names, monthly.values, color=colors)
    for i, v in enumerate(monthly.values):
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel(f"{col}, ug/m3")
    ax.set_title(f"Average {col} by calendar month (teal = winter, orange = summer)", fontsize=10)
    ax.grid(axis="x", visible=False)
    return _save(fig, filename)


def plot_interactive(df):
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
    ensure_dirs()
    describe_data(df).to_csv(config.TABLE_DIR / "descriptive_stats.csv")
    iqr_outliers(df).to_csv(config.TABLE_DIR / "iqr_outliers.csv", index=False)
    correlation_matrix(df).to_csv(config.TABLE_DIR / "correlation.csv")
    figures = [plot_boxplot_by_station(df), plot_monthly_by_area(df), plot_season_by_station(df),
               plot_hourly_profile(df), plot_correlation(df), plot_mean_vs_median(df), plot_monthly_mean(df), plot_interactive(df)]
    return [f for f in figures if f is not None]
