
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "adj_close.csv"
OUT_DIR = PROJECT_ROOT / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "adj_close_clean.csv"

def clean_data():
    print(f"[clean] Reading {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    df.set_index("date", inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna()
    df = df[(df > 0).all(axis=1)]
    after = len(df)

    print(f"[clean] Dropped {before - after} bad rows (NaN or <=0)")

    corr_matrix = df.corr()

    print(f"[clean] Final shape: {df.shape[0]} rows × {df.shape[1]} tickers")
    print(f"[clean] Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"[clean] Example correlations:\n{corr_matrix.iloc[:4, :4].round(3)}")

    df.to_csv(OUT_PATH)
    print(f"[clean] Saved cleaned data to {OUT_PATH}")


if __name__ == "__main__":
    clean_data()
