import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PAIR = "MA_V"
FILE = PROCESSED_DIR / f"paper_results_{PAIR}.csv"

df = pd.read_csv(FILE, parse_dates=["date"]).set_index("date")


plt.figure(figsize=(12,4))
plt.plot(df["equity"], label="Equity")
plt.title(f"Equity Curve – {PAIR}")
plt.ylabel("Portfolio Value ($)")
plt.tight_layout()
plt.savefig(PROCESSED_DIR / f"equity_curve_{PAIR}.png")
plt.close()


cummax = df["equity"].cummax()
drawdown = df["equity"] / cummax - 1
plt.figure(figsize=(12,4))
plt.plot(drawdown, color="red")
plt.title(f"Drawdown – {PAIR}")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig(PROCESSED_DIR / f"drawdown_{PAIR}.png")
plt.close()


plt.figure(figsize=(12,4))
plt.plot(df["zscore"], label="Z-score")
plt.axhline(2.0, color="black", linestyle="--", label="Entry +2")
plt.axhline(-2.0, color="black", linestyle="--", label="Entry -2")
plt.axhline(0.5, color="gray", linestyle=":", label="Exit band")
plt.axhline(-0.5, color="gray", linestyle=":")
plt.title(f"Z-score with Entry/Exit Bands – {PAIR}")
plt.legend()
plt.tight_layout()
plt.savefig(PROCESSED_DIR / f"zscore_signals_{PAIR}.png")
plt.close()


plt.figure(figsize=(6,4))
df["ret"].hist(bins=50)
plt.title(f"Daily Return Distribution – {PAIR}")
plt.tight_layout()
plt.savefig(PROCESSED_DIR / f"returns_hist_{PAIR}.png")
plt.close()

print("Saved plots to:", PROCESSED_DIR)
