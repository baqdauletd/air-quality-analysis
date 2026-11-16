"""Streamlit dashboard.   Run:  streamlit run app.py   (after  python main.py --all)"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import db

st.set_page_config(page_title="Beijing Air Quality", layout="wide")


@st.cache_data
def load_daily():
    """Daily means per station straight from SQLite (SQL aggregation keeps the app fast)."""
    return db.run_query("""
        SELECT s.name AS station, s.area_type, date(p.ts) AS date, p.year, p.month,
               AVG(p.pm25) AS pm25, AVG(p.pm10) AS pm10, AVG(p.so2) AS so2, AVG(p.no2) AS no2,
               AVG(p.co) AS co, AVG(p.o3) AS o3,
               AVG(w.temp) AS temp, AVG(w.wspm) AS wspm
        FROM pollution p JOIN stations s USING (station_id) JOIN weather w USING (reading_id)
        GROUP BY s.name, date(p.ts)""").assign(date=lambda d: pd.to_datetime(d["date"]))


@st.cache_data
def load_missing():
    """Hours and missing hours per station and month (after cleaning), counted in SQLite."""
    cols = ["pm25", "pm10", "so2", "no2", "co", "o3"]
    sums = ", ".join(f"SUM(p.{c} IS NULL) AS {c}" for c in cols)
    return db.run_query(f"""
        SELECT s.name AS station, p.year, p.month, COUNT(*) AS hours, {sums}
        FROM pollution p JOIN stations s USING (station_id)
        GROUP BY s.name, p.year, p.month""")


@st.cache_data
def load_missing_summary():
    """Missing values per column before and after cleaning (written by main.py)."""
    path = config.TABLE_DIR / "missing_values.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


if not config.DB_PATH.exists():
    st.error("Database not found. Run `python main.py --all` first.")
    st.stop()

daily = load_daily()
st.title("Beijing air quality, 2013-2017")

with st.sidebar:
    st.header("Filters")
    years = st.slider("Years", int(daily.year.min()), int(daily.year.max()),
                      (int(daily.year.min()), int(daily.year.max())))
    area = st.multiselect("Area type", sorted(daily.area_type.unique()), default=sorted(daily.area_type.unique()))
    stations_all = sorted(daily[daily.area_type.isin(area)].station.unique())
    stations = st.multiselect("Stations", stations_all, default=stations_all)
    pollutant = st.selectbox("Pollutant", ["pm25", "pm10", "so2", "no2", "co", "o3"])
    freq = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"], index=2)

view = daily[daily.year.between(*years) & daily.station.isin(stations)]
if view.empty:
    st.warning("No data for the selected filters.")
    st.stop()

rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq]
series = (view.set_index("date").groupby("station")[pollutant].resample(rule).mean().reset_index())

c1, c2, c3 = st.columns(3)
c1.metric(f"Mean {pollutant.upper()}", f"{view[pollutant].mean():.1f}")
c2.metric("Highest daily mean", f"{view[pollutant].max():.0f}")
c3.metric("Stations selected", len(stations))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time series", "Seasons", "Stations", "Weather link", "Data quality"])
with tab1:
    st.plotly_chart(px.line(series, x="date", y=pollutant, color="station"), use_container_width=True)
with tab2:
    view = view.assign(season=view.month.map(config.SEASONS))
    seas = view.groupby(["season", "area_type"])[pollutant].mean().reset_index()
    st.plotly_chart(px.bar(seas, x="season", y=pollutant, color="area_type", barmode="group",
                           category_orders={"season": config.SEASON_ORDER}), use_container_width=True)
with tab3:
    stats = view.groupby("station")[pollutant].agg(["mean", "median"]).sort_index()   # alphabetical
    order = list(stats.index)                # same station order in both charts
    layout = dict(height=450, margin=dict(l=70, r=20, t=80, b=90), xaxis_tickangle=-45,
                  legend=dict(orientation="h", y=1.12, x=0))   # legend above, so both plots have the same width
    box = px.box(view, x="station", y=pollutant, color="area_type", category_orders={"station": order})
    box.update_layout(title=f"Distribution of daily {pollutant.upper()} by station", **layout)
    st.plotly_chart(box, use_container_width=True)
    dumbbell = go.Figure()
    for name, row in stats.iterrows():       # grey line between median and mean = size of the gap
        dumbbell.add_trace(go.Scatter(x=[name, name], y=[row["median"], row["mean"]], mode="lines",
                                      line=dict(color="#9aa5ad", width=3), showlegend=False, hoverinfo="skip"))
    dumbbell.add_trace(go.Scatter(x=stats.index, y=stats["median"], mode="markers", name="median",
                                  marker=dict(color="#E07A2F", size=11)))
    dumbbell.add_trace(go.Scatter(x=stats.index, y=stats["mean"], mode="markers", name="mean",
                                  marker=dict(color="#0E7C86", size=11)))
    dumbbell.update_layout(title=f"Mean vs median of daily {pollutant.upper()} by station",
                           yaxis_title=f"{pollutant.upper()} (daily means)",
                           xaxis=dict(categoryorder="array", categoryarray=order), **layout)
    st.plotly_chart(dumbbell, use_container_width=True)
    by_season = (view.assign(season=view.month.map(config.SEASONS))
                 .groupby(["station", "season"])[pollutant].mean().reset_index())
    st.plotly_chart(px.bar(by_season, x="station", y=pollutant, color="season", barmode="group",
                           category_orders={"season": config.SEASON_ORDER},
                           title=f"Seasonal mean {pollutant.upper()} by station"),
                    use_container_width=True)
with tab4:
    weather_names = {"temp": "Temperature (temp)", "wspm": "Wind speed (wspm)"}
    x = st.selectbox("Weather variable", list(weather_names), format_func=weather_names.get)
    valid = view.dropna(subset=[pollutant, x])          # rows with both values present
    sample = valid.sample(min(4000, len(valid)), random_state=1) if len(valid) else valid
    st.plotly_chart(px.scatter(sample, x=x, y=pollutant, color="area_type", opacity=0.5), use_container_width=True)
with tab5:
    summary = load_missing_summary()
    if summary is not None:
        long = (summary[["percent_before", "percent_after"]]
                .rename(columns={"percent_before": "before cleaning", "percent_after": "after cleaning"})
                .reset_index().rename(columns={"index": "column"})
                .melt(id_vars="column", var_name="stage", value_name="missing, %"))
        st.plotly_chart(px.bar(long, x="column", y="missing, %", color="stage", barmode="group"),
                        use_container_width=True)
        st.dataframe(summary.rename(columns={"missing_before": "missing (before)", "percent_before": "% (before)",
                                             "missing_after": "missing (after)", "percent_after": "% (after)"}),
                     use_container_width=True)

    miss = load_missing()
    miss = miss[miss.year.between(*years) & miss.station.isin(stations)]
    st.subheader("After cleaning, by stations")
    by_station = miss.groupby("station")[[pollutant, "hours"]].sum()
    by_station["missing, %"] = 100 * by_station[pollutant] / by_station["hours"]
    st.plotly_chart(px.bar(by_station.reset_index().sort_values("missing, %", ascending=False),
                           x="station", y="missing, %"), use_container_width=True)

    miss = miss.assign(period=miss.year.astype(str) + "-" + miss.month.astype(str).str.zfill(2))
    grid = miss.groupby(["station", "period"])[[pollutant, "hours"]].sum()
    grid["missing, %"] = 100 * grid[pollutant] / grid["hours"]
    heat = grid["missing, %"].unstack("period")
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Reds",
                              labels={"x": "month", "y": "station", "color": "missing, %"},
                              title="Missing data - stations x months"),
                    use_container_width=True)
