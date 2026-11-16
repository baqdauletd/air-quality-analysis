# Air Quality Across Monitoring Sites in Beijing

Final project, Programming for Data Analysis and Database (Applied AI, Master's, Astana IT University).

Topic 2, Cluster 1: seasonal PM2.5/PM10 variation, site comparison and weather influence.

**Team:** Bakdaulet Dauletov and Ainur Taskynova (AAI-2603M)

## Data
Beijing Multi-Site Air Quality Data (UCI, dataset 501): hourly data from 12 stations, 2013-03-01 to 2017-02-28, 420,768 rows.
Source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
The 12 CSV files (31 MB) are in `data/raw/`.

## What the project does
| Step | Module | Result |
|---|---|---|
| Load (chunks, compact dtypes), clean, fill gaps up to 6 h | `data_loader.py` | `data/processed/clean_air_quality.csv` (+ Parquet if pyarrow installed) |
| SQLite schema (4 tables), ER diagram, 10 SQL queries | `db.py` | `data/processed/air_quality.db`, `outputs/tables/q*.csv`, `outputs/figures/er_diagram.png` |
| EDA: statistics, IQR outliers, correlation, figures, interactive Plotly chart | `analysis.py` | `outputs/figures`, `outputs/tables` |
| Resampling, moving averages, seasonal decomposition, site comparison | `timeseries.py` | decomposition and moving-average figures |
| Next-day forecast: naive and seasonal-naive baselines, moving average, Linear Regression (MAE) | `forecast.py` | `outputs/tables/forecast_metrics_citywide.csv` |
| Comparison of processing approaches (pandas, SQL, Parquet) | `benchmarks.py` | `outputs/tables/benchmark_*.csv` |
| Console menu | `main.py` | |
| Streamlit dashboard with filters (year, area type, station, pollutant) | `app.py` | |

## How to run
Python 3.10 or newer.
```bash
pip install -r requirements.txt
python main.py --all        # runs every step, writes tables and figures to outputs/
python main.py              # interactive menu (options 1-9)
streamlit run app.py        # dashboard (run main.py --all first, it needs the database)
```
Install all packages from `requirements.txt` to get everything.
The dashboard needs `plotly` and `streamlit`.

## Notes
- Years 2013 and 2017 are partial (March-December 2013, January-February 2017).
- The written report is in `report/`, the slides in `presentation/`.

## Project contributions
- Bakdaulet Dauletov: loading and cleaning, SQLite and SQL, ER diagram, processing comparison, repository.
- Ainur Taskynova: EDA, time series, visualizations, forecasting, dashboard.
- Both: report, presentation, testing.
