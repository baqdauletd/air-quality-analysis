"""Time-series tools: resampling, moving averages and seasonal decomposition."""
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import ensure_dirs, has_module


def resample_series(df, col="PM2.5", freq="MS", station=None):
    """Mean of `col` resampled to `freq` (D, W, MS...). station=None -> all stations."""
    data = df if station is None else df[df["station"].astype(str) == station]
    return data.set_index("ts")[col].resample(freq).mean()


def moving_average(series, window):
    return series.rolling(window, min_periods=max(1, window // 2)).mean()


@dataclass
class Decomposition:
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    method: str


def classical_decompose(series, period=12):
    """Classical additive decomposition (same algorithm as statsmodels.seasonal_decompose)."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    if period % 2 == 0:
        weights = np.r_[0.5, np.ones(period - 1), 0.5] / period
    else:
        weights = np.ones(period) / period
    trend = np.full(n, np.nan)
    half = len(weights) // 2
    trend[half:n - half] = np.convolve(values, weights, mode="valid")
    detrended = values - trend
    seasonal_means = np.array([np.nanmean(detrended[i::period]) for i in range(period)])
    seasonal_means -= seasonal_means.mean()
    seasonal = np.resize(seasonal_means, n)
    resid = values - trend - seasonal
    idx = series.index
    return Decomposition(series, pd.Series(trend, idx), pd.Series(seasonal, idx),
                         pd.Series(resid, idx), "classical (own implementation)")


def decompose(series, period=12):
    """Seasonal decomposition. Uses statsmodels when installed, otherwise own implementation."""
    series = series.interpolate(limit_direction="both")
    if has_module("statsmodels"):
        from statsmodels.tsa.seasonal import seasonal_decompose
        res = seasonal_decompose(series, model="additive", period=period)
        return Decomposition(series, res.trend, res.seasonal, res.resid, "statsmodels.seasonal_decompose")
    return classical_decompose(series, period)


def seasonal_strength(dec):
    """Share of the de-trended variance explained by the seasonal part (0..1)."""
    resid = dec.resid.dropna()
    seas = dec.seasonal.loc[resid.index]
    return float(max(0.0, 1 - resid.var() / (resid + seas).var()))


def plot_decomposition(dec, title, filename):
    ensure_dirs()
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, (name, s) in zip(axes, [("Observed", dec.observed), ("Trend", dec.trend),
                                    ("Seasonal", dec.seasonal), ("Residual", dec.resid)]):
        ax.plot(s.index, s.values, color="#1f5f99")
        ax.set_ylabel(name)
    axes[0].set_title(f"{title}  [{dec.method}]")
    path = config.FIG_DIR / filename
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def station_seasonality_table(df, pollutants=("PM2.5", "PM10")):
    """Decompose each station's monthly series and compare seasonal amplitude/strength."""
    rows = []
    for station in sorted(df["station"].astype(str).unique()):
        for col in pollutants:
            dec = decompose(resample_series(df, col, "MS", station))
            amp = float(dec.seasonal.max() - dec.seasonal.min())
            trend = dec.trend.dropna()
            rows.append({"station": station, "pollutant": col,
                         "area_type": config.AREA_TYPE[station],
                         "seasonal_amplitude": round(amp, 1),
                         "seasonal_strength": round(seasonal_strength(dec), 3),
                         "trend_change": round(float(trend.iloc[-1] - trend.iloc[0]), 1)})
    return pd.DataFrame(rows)


def run_timeseries(df):
    """Decomposition of the city-wide PM2.5 + comparison table of all stations."""
    ensure_dirs()
    city = resample_series(df, "PM2.5", "MS")
    dec = decompose(city, 12)
    path = plot_decomposition(dec, "City-wide monthly PM2.5", "07_decomposition_city_pm25.png")
    seasonal_profile = dec.seasonal.groupby(dec.seasonal.index.month).first().round(1)
    seasonal_profile.to_csv(config.TABLE_DIR / "seasonal_component_by_month.csv", header=["seasonal_effect"])
    table = station_seasonality_table(df)
    table.to_csv(config.TABLE_DIR / "station_seasonality.csv", index=False)

    ma = pd.DataFrame({"daily": resample_series(df, "PM2.5", "D")})
    ma["ma7"] = moving_average(ma["daily"], 7)
    ma["ma30"] = moving_average(ma["daily"], 30)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ma.index, ma["daily"], alpha=0.3, label="daily mean")
    ax.plot(ma.index, ma["ma7"], label="7-day MA")
    ax.plot(ma.index, ma["ma30"], label="30-day MA", linewidth=2)
    ax.set_title("City-wide daily PM2.5 and moving averages")
    ax.set_ylabel("PM2.5, ug/m3")
    ax.legend()
    ma_path = config.FIG_DIR / "08_moving_averages.png"
    fig.savefig(ma_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"method": dec.method, "strength": seasonal_strength(dec), "figures": [path, ma_path],
            "table": table, "seasonal_profile": seasonal_profile}
