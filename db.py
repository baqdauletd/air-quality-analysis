import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import config
from utils import ensure_dirs

SCHEMA = """
DROP TABLE IF EXISTS weather;
DROP TABLE IF EXISTS pollution;
DROP TABLE IF EXISTS wind_directions;
DROP TABLE IF EXISTS stations;

CREATE TABLE stations (
    station_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    area_type  TEXT NOT NULL
);
CREATE TABLE wind_directions (
    wd_id INTEGER PRIMARY KEY,
    code  TEXT NOT NULL UNIQUE
);
CREATE TABLE pollution (
    reading_id INTEGER PRIMARY KEY,
    station_id INTEGER NOT NULL REFERENCES stations(station_id),
    ts         TEXT NOT NULL,
    year       INTEGER, month INTEGER,
    pm25 REAL, pm10 REAL, so2 REAL, no2 REAL, co REAL, o3 REAL
);
CREATE TABLE weather (
    reading_id INTEGER PRIMARY KEY REFERENCES pollution(reading_id),
    wd_id      INTEGER REFERENCES wind_directions(wd_id),
    temp REAL, pres REAL, dewp REAL, rain REAL, wspm REAL
);
CREATE INDEX idx_pollution_station_ts ON pollution(station_id, ts);
CREATE INDEX idx_pollution_year_month ON pollution(year, month);
"""

QUERIES = {
    "q01_station_averages": (
        "JOIN + GROUP BY: average pollution per station",
        """SELECT s.name, s.area_type, ROUND(AVG(p.pm25),1) AS avg_pm25,
                  ROUND(AVG(p.pm10),1) AS avg_pm10, COUNT(*) AS readings
           FROM pollution p JOIN stations s USING (station_id)
           GROUP BY s.station_id ORDER BY avg_pm25 DESC"""),
    "q02_stations_above_average": (
        "GROUP BY + HAVING + subquery: stations dirtier than the city average",
        """SELECT s.name, ROUND(AVG(p.pm25),1) AS avg_pm25
           FROM pollution p JOIN stations s USING (station_id)
           GROUP BY s.name
           HAVING AVG(p.pm25) > (SELECT AVG(pm25) FROM pollution)
           ORDER BY avg_pm25 DESC"""),
    "q03_monthly_citywide": (
        "GROUP BY on a date expression: monthly mean PM2.5 for the whole city",
        """SELECT strftime('%Y-%m', ts) AS year_month, ROUND(AVG(pm25),1) AS avg_pm25
           FROM pollution GROUP BY year_month ORDER BY year_month"""),
    "q04_season_by_area_type": (
        "JOIN + CASE: seasonal PM2.5 for urban / suburban / rural stations",
        """SELECT s.area_type,
                  CASE WHEN p.month IN (12,1,2) THEN 'Winter'
                       WHEN p.month IN (3,4,5)  THEN 'Spring'
                       WHEN p.month IN (6,7,8)  THEN 'Summer'
                       ELSE 'Autumn' END AS season,
                  ROUND(AVG(p.pm25),1) AS avg_pm25
           FROM pollution p JOIN stations s USING (station_id)
           GROUP BY s.area_type, season ORDER BY s.area_type, avg_pm25 DESC"""),
    "q05_heavy_pollution_days": (
        "CTE: share of days with daily mean PM2.5 above 150 per station",
        """WITH daily AS (
               SELECT station_id, date(ts) AS d, AVG(pm25) AS pm
               FROM pollution GROUP BY station_id, date(ts))
           SELECT s.name, COUNT(*) AS days,
                  SUM(CASE WHEN pm > 150 THEN 1 ELSE 0 END) AS heavy_days,
                  ROUND(100.0 * SUM(CASE WHEN pm > 150 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_heavy
           FROM daily JOIN stations s USING (station_id)
           WHERE pm IS NOT NULL GROUP BY s.name ORDER BY pct_heavy DESC"""),
    "q06_rank_stations_by_year": (
        "Window function RANK(): station ranking inside each year",
        """WITH yearly AS (
               SELECT station_id, year, AVG(pm25) AS avg_pm
               FROM pollution GROUP BY station_id, year)
           SELECT year, s.name, ROUND(avg_pm,1) AS avg_pm25,
                  RANK() OVER (PARTITION BY year ORDER BY avg_pm DESC) AS rank_in_year
           FROM yearly JOIN stations s USING (station_id)
           ORDER BY year, rank_in_year"""),
    "q07_moving_average_dongsi": (
        "Window function AVG OVER: 7-day moving average of daily PM2.5 (Dongsi)",
        """WITH daily AS (
               SELECT date(ts) AS d, AVG(pm25) AS pm FROM pollution
               WHERE station_id = (SELECT station_id FROM stations WHERE name = 'Dongsi')
               GROUP BY date(ts))
           SELECT d, ROUND(pm,1) AS daily_pm25,
                  ROUND(AVG(pm) OVER (ORDER BY d ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),1) AS ma7
           FROM daily ORDER BY d"""),
    "q08_pm25_by_wind_direction": (
        "Three-table JOIN: PM2.5 by wind direction",
        """SELECT w.code AS wind_dir, COUNT(*) AS readings, ROUND(AVG(p.pm25),1) AS avg_pm25,
                  ROUND(AVG(x.wspm),2) AS avg_wind_speed
           FROM pollution p JOIN weather x USING (reading_id)
                            JOIN wind_directions w ON w.wd_id = x.wd_id
           GROUP BY w.code ORDER BY avg_pm25 DESC"""),
    "q09_rain_effect": (
        "JOIN + CASE: PM2.5 in rain vs. dry hours",
        """SELECT CASE WHEN x.rain > 0 THEN 'Rain' ELSE 'No rain' END AS condition,
                  COUNT(*) AS readings, ROUND(AVG(p.pm25),1) AS avg_pm25
           FROM pollution p JOIN weather x USING (reading_id)
           GROUP BY condition"""),
    "q10_year_over_year_change": (
        "Window function LAG(): change of monthly PM2.5 versus the previous year",
        """WITH monthly AS (
               SELECT year, month, AVG(pm25) AS pm FROM pollution
               WHERE year BETWEEN 2014 AND 2016 GROUP BY year, month)
           SELECT year, month, ROUND(pm,1) AS avg_pm25,
                  ROUND(pm - LAG(pm) OVER (PARTITION BY month ORDER BY year),1) AS change_vs_prev_year
           FROM monthly ORDER BY month, year"""),
}


def connect(db_path=config.DB_PATH):
    if not db_path.exists():
        raise FileNotFoundError("Database not found. Run option 2 (build SQLite database) first.")
    return sqlite3.connect(db_path)


def build_database(df, db_path=config.DB_PATH):
    """Create the schema and load the cleaned DataFrame into SQLite."""
    ensure_dirs()
    stations = pd.DataFrame({"name": sorted(df["station"].astype(str).unique())})
    stations["station_id"] = range(1, len(stations) + 1)
    stations["area_type"] = stations["name"].map(config.AREA_TYPE)
    winds = pd.DataFrame({"code": sorted(df["wd"].dropna().astype(str).unique())})
    winds["wd_id"] = range(1, len(winds) + 1)

    work = df.reset_index(drop=True)
    pollution = pd.DataFrame({
        "reading_id": range(1, len(work) + 1),
        "station_id": work["station"].astype(str).map(dict(zip(stations.name, stations.station_id))),
        "ts": work["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "year": work["year"].astype(int), "month": work["month"].astype(int),
        "pm25": work["PM2.5"], "pm10": work["PM10"], "so2": work["SO2"],
        "no2": work["NO2"], "co": work["CO"], "o3": work["O3"],
    })
    weather = pd.DataFrame({
        "reading_id": pollution["reading_id"],
        "wd_id": work["wd"].astype(str).map(dict(zip(winds.code, winds.wd_id))),
        "temp": work["TEMP"], "pres": work["PRES"], "dewp": work["DEWP"],
        "rain": work["RAIN"], "wspm": work["WSPM"],
    })

    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA)
        stations[["station_id", "name", "area_type"]].to_sql("stations", con, if_exists="append", index=False)
        winds[["wd_id", "code"]].to_sql("wind_directions", con, if_exists="append", index=False)
        pollution.to_sql("pollution", con, if_exists="append", index=False, chunksize=50000)
        weather.to_sql("weather", con, if_exists="append", index=False, chunksize=50000)
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("stations", "wind_directions", "pollution", "weather")}
    return counts


def run_query(sql, db_path=config.DB_PATH):
    """Run a SQL query and return the result as a Pandas DataFrame."""
    con = connect(db_path)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def run_all_queries(db_path=config.DB_PATH, save=True):
    """Run the 10 analytics queries; save every result to outputs/tables."""
    ensure_dirs()
    results = {}
    for name, (title, sql) in QUERIES.items():
        results[name] = run_query(sql, db_path)
        if save:
            results[name].to_csv(config.TABLE_DIR / f"{name}.csv", index=False)
    return results


def _draw_table(ax, x, y, title, columns, width=3.1):
    row_h = 0.32
    height = row_h * (len(columns) + 1)
    ax.add_patch(FancyBboxPatch((x, y - height), width, height, boxstyle="round,pad=0.02",
                                fc="#f7f7f7", ec="#333333"))
    ax.add_patch(FancyBboxPatch((x, y - row_h), width, row_h, boxstyle="round,pad=0.02",
                                fc="#cfe3f5", ec="#333333"))
    ax.text(x + width / 2, y - row_h / 2, title, ha="center", va="center", fontweight="bold")
    for i, col in enumerate(columns, start=1):
        ax.text(x + 0.1, y - row_h * i - row_h / 2, col, va="center", fontsize=8.5, family="monospace")
    return {"x": x, "y": y, "w": width, "h": height}


def draw_er_diagram(path=None):
    """Draw the ER diagram of the database with matplotlib."""
    ensure_dirs()
    path = path or config.FIG_DIR / "er_diagram.png"
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    st = _draw_table(ax, 0.3, 6.2, "stations", ["PK station_id", "   name", "   area_type"])
    po = _draw_table(ax, 4.0, 6.2, "pollution", [
        "PK reading_id", "FK station_id", "   ts", "   year, month",
        "   pm25, pm10, so2", "   no2, co, o3"])
    we = _draw_table(ax, 7.7, 6.2, "weather", [
        "PK/FK reading_id", "FK wd_id", "   temp, pres, dewp", "   rain, wspm"])
    wd = _draw_table(ax, 7.7, 3.2, "wind_directions", ["PK wd_id", "   code"])

    def link(x1, y1, x2, y2, l1, l2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-", lw=1.5))
        ax.text(x1 + (0.15 if x2 > x1 else -0.15), y1 + 0.12, l1, fontsize=10, fontweight="bold")
        ax.text(x2 - (0.15 if x2 > x1 else -0.15), y2 + 0.12, l2, fontsize=10, fontweight="bold", ha="right")

    link(st["x"] + st["w"], 5.3, po["x"], 5.3, "1", "N")
    link(po["x"] + po["w"], 5.3, we["x"], 5.3, "1", "1")
    ax.annotate("", xy=(we["x"] + 1.5, 3.2), xytext=(we["x"] + 1.5, we["y"] - we["h"]),
                arrowprops=dict(arrowstyle="-", lw=1.5))
    ax.text(we["x"] + 1.65, 3.3, "1", fontsize=10, fontweight="bold")
    ax.text(we["x"] + 1.65, we["y"] - we["h"] - 0.3, "N", fontsize=10, fontweight="bold")
    ax.set_title("ER diagram - Beijing air quality database (SQLite)", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
