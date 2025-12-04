

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "adj_close_clean.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "hedge_results.csv"

ROLLING_Z_WINDOW = 60 


def _fit_ols(y: pd.Series, x: pd.Series):

    df = pd.concat({"y": y, "x": x}, axis=1).dropna()
    X = sm.add_constant(df["x"]) 
    model = sm.OLS(df["y"], X)
    res = model.fit()

    alpha = float(res.params["const"])
    beta = float(res.params["x"])
    r2 = float(res.rsquared)
    resid = df["y"] - (alpha + beta * df["x"])
    return alpha, beta, r2, resid


def main(y_ticker: str | None = None, x_ticker: str | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Compute hedge ratio for a pair")
    parser.add_argument("--y", type=str, help="Ticker Y (dependent variable)")
    parser.add_argument("--x", type=str, help="Ticker X (independent variable)")
    args = parser.parse_args()

    if args.y:
        y_ticker = args.y.upper()
    if args.x:
        x_ticker = args.x.upper()

    if not y_ticker or not x_ticker:
        raise ValueError("You must provide tickers using --y and --x")

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Clean file not found: {CLEAN_PATH}. Run: python -m src.data.clean first."
        )

    df = pd.read_csv(CLEAN_PATH, parse_dates=["date"]).set_index("date")

    df = df[[y_ticker, x_ticker]].dropna()

    alpha, beta, r2, resid = _fit_ols(df[y_ticker], df[x_ticker])
    spread = df[y_ticker] - beta * df[x_ticker]
    spread.name = "spread"

    mu = float(spread.mean())
    sigma = float(spread.std(ddof=1))

    if sigma == 0 or np.isnan(sigma):
        raise RuntimeError("Spread std is zero/NaN; cannot compute Z.")

    z_full = (spread - mu) / sigma
    z_full.name = "zscore_full"

    out = pd.concat([df, spread, z_full], axis=1)

    PAIR = f"{y_ticker}_{x_ticker}"
    out_path = OUT_DIR / f"hedge_results_{PAIR}.csv"
    out.to_csv(out_path)

    print(f"[hedge] ===== OLS results ({PAIR}) =====")
    print(f"[hedge] beta: {beta:.4f}, alpha: {alpha:.4f}, R2: {r2:.3f}")
    print(f"[hedge] mean={mu:.4f}, std={sigma:.4f}")
    print(f"[hedge] saved → {out_path}  (rows={out.shape[0]})")

    return out_path


if __name__ == "__main__":
    main()
