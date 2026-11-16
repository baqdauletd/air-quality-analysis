from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DB_PATH = PROCESSED_DIR / "air_quality.db"
CLEAN_CSV = PROCESSED_DIR / "clean_air_quality.csv"
CLEAN_PARQUET = PROCESSED_DIR / "clean_air_quality.parquet"

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC_COLS = POLLUTANTS + WEATHER

AREA_TYPE = {
    "Aotizhongxin": "Urban",
    "Dongsi": "Urban",
    "Guanyuan": "Urban",
    "Gucheng": "Urban",
    "Nongzhanguan": "Urban",
    "Tiantan": "Urban",
    "Wanliu": "Urban",
    "Wanshouxigong": "Urban",
    "Changping": "Suburban",
    "Shunyi": "Suburban",
    "Dingling": "Rural",
    "Huairou": "Rural",
}

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer",
           9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

MAX_GAP_HOURS = 6        # for interpolation
HEAVY_POLLUTION = 150    # daily mean PM2.5 (ug/m3) treated as a heavy-pollution day
TEST_START = "2016-03-01"  # forecast test period - last 12 months
