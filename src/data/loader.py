
from pathlib import Path
import yaml
import pandas as pd


import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_prices():
    cfg = yaml.safe_load(open(CONFIG_PATH, "r", encoding="utf-8"))
    tickers = [t.strip().upper() for t in cfg["tickers"]]
    start = cfg["start_date"]
    end = cfg["end_date"]
    interval = cfg.get("interval", "1d")

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,   
        group_by="column",
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError("No data returned.")

    if isinstance(df.columns, pd.MultiIndex):
        adj = df["Adj Close"].copy()
    else:
        adj = df.copy()
        adj.columns = tickers[:1]

    adj.index.name = "date"
    adj = adj[tickers] 
    adj = adj.sort_index()

    out_path = RAW_DIR / "adj_close.csv"
    adj.to_csv(out_path, index=True)


    print(f"[loader] saved {out_path} with shape {adj.shape}")
    print(f"[loader] dates: {adj.index.min().date()} → {adj.index.max().date()}")
    return out_path


if __name__ == "__main__":
    load_prices()
