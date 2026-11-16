"""Next-day PM2.5 forecast: baselines versus Linear Regression (metric: MAE)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import ensure_dirs, measure


def build_daily_dataset(df, station=None):
    """Daily means of PM2.5 and weather, plus lag features (weather is shifted by one day)."""
    data = df if station is None else df[df["station"].astype(str) == station]
    daily = data.set_index("ts")[["PM2.5", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]].resample("D").mean()
    daily = daily.interpolate(limit=3)
    out = pd.DataFrame({"y": daily["PM2.5"]})
    for lag in (1, 2, 3, 7):
        out[f"lag{lag}"] = daily["PM2.5"].shift(lag)
    out["roll7"] = daily["PM2.5"].shift(1).rolling(7).mean()
    out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12)
    for col in ("TEMP", "PRES", "DEWP", "RAIN", "WSPM"):
        out[f"{col}_prev"] = daily[col].shift(1)
    return out.dropna()


FEATURES_LAGS = ["lag1", "lag2", "lag3", "lag7", "roll7"]
FEATURES_ALL = FEATURES_LAGS + ["month_sin", "month_cos", "TEMP_prev", "PRES_prev",
                                "DEWP_prev", "RAIN_prev", "WSPM_prev"]


class LinearRegressionNP:
    """Ordinary least squares with standardised inputs (NumPy only)."""

    def fit(self, X, y):
        self.mean, self.std = X.mean(0), X.std(0) + 1e-9
        A = np.c_[np.ones(len(X)), (X - self.mean) / self.std]
        self.coef = np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), (X - self.mean) / self.std] @ self.coef


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def run_models(data, test_start=config.TEST_START):
    """Fit/evaluate all models; returns (metrics table, predictions DataFrame)."""
    train, test = data[data.index < test_start], data[data.index >= test_start]
    preds, rows = {}, []

    def evaluate(name, func):
        pred, seconds, peak = measure(func)
        preds[name] = pred
        rows.append({"model": name, "MAE": round(mae(test["y"], pred), 2),
                     "RMSE": round(rmse(test["y"], pred), 2),
                     "time_ms": round(seconds * 1000, 2), "peak_memory_MB": round(peak, 3)})

    evaluate("Baseline: naive (yesterday)", lambda: test["lag1"].to_numpy())
    evaluate("Baseline: seasonal naive (same weekday last week)", lambda: test["lag7"].to_numpy())
    evaluate("Moving average (7 days)", lambda: test["roll7"].to_numpy())

    def lr(features):
        model = LinearRegressionNP().fit(train[features].to_numpy(), train["y"].to_numpy())
        return model.predict(test[features].to_numpy())

    evaluate("Linear Regression (lags only)", lambda: lr(FEATURES_LAGS))
    evaluate("Linear Regression (lags + weather + month)", lambda: lr(FEATURES_ALL))
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    pred_df = pd.DataFrame(preds, index=test.index)
    pred_df.insert(0, "actual", test["y"])
    return table, pred_df


def plot_forecast(pred_df, best_name, filename="09_forecast_vs_actual.png"):
    ensure_dirs()
    last = pred_df.iloc[-120:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(last.index, last["actual"], label="actual", color="black")
    ax.plot(last.index, last["Baseline: naive (yesterday)"], label="naive", alpha=0.6)
    ax.plot(last.index, last[best_name], label=best_name, linewidth=2)
    ax.set_title("Next-day PM2.5 forecast - last 120 days of the test period")
    ax.set_ylabel("PM2.5, ug/m3")
    ax.legend(fontsize=8)
    path = config.FIG_DIR / filename
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def run_forecast(df, station=None):
    """Forecast experiment. station=None -> city-wide average of the 12 stations."""
    data = build_daily_dataset(df, station)
    table, pred_df = run_models(data)
    suffix = "citywide" if station is None else station.lower()
    table.to_csv(config.TABLE_DIR / f"forecast_metrics_{suffix}.csv", index=False)
    figure = plot_forecast(pred_df, table.loc[0, "model"])
    return table, figure
