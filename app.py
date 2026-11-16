"""Streamlit dashboard.   Run:  streamlit run app.py   (after  python main.py --all)"""
import pandas as pd
import plotly.express as px
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

tab1, tab2, tab3, tab4 = st.tabs(["Time series", "Seasons", "Stations", "Weather link"])
with tab1:
    st.plotly_chart(px.line(series, x="date", y=pollutant, color="station"), use_container_width=True)
with tab2:
    view = view.assign(season=view.month.map(config.SEASONS))
    seas = view.groupby(["season", "area_type"])[pollutant].mean().reset_index()
    st.plotly_chart(px.bar(seas, x="season", y=pollutant, color="area_type", barmode="group",
                           category_orders={"season": config.SEASON_ORDER}), use_container_width=True)
with tab3:
    st.plotly_chart(px.box(view, x="station", y=pollutant, color="area_type"), use_container_width=True)
with tab4:
    x = st.selectbox("Weather variable", ["temp", "wspm"])
    sample = view.dropna(subset=[pollutant, x]).sample(min(4000, len(view)), random_state=1)
    st.plotly_chart(px.scatter(sample, x=x, y=pollutant, color="area_type", opacity=0.5), use_container_width=True)
