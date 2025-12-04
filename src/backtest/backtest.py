from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS_PER_YEAR = 252
TURNOVER_COST = 0.0002

def _estimate_beta(y: pd.Series, x: pd.Series) -> float:
    df = pd.concat({"y": y, "x": x}, axis=1).dropna()
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit()
    return float(model.params["x"])


def _compute_portfolio_returns(df: pd.DataFrame, y_col: str, x_col: str, beta: float,) -> pd.Series:
    y = df[y_col]
    x = df[x_col]
    pos = df["position"]

    y_ret = y.pct_change()
    x_ret = x.pct_change()

    pos_lag = pos.shift(1).fillna(0)

    spread_ret = y_ret - beta * x_ret

    turnover = pos_lag.diff().abs().fillna(0)

    cost = TURNOVER_COST * turnover

    strat_ret = pos_lag * spread_ret - cost
    strat_ret.name = "strategy_return"
    return strat_ret




def _compute_performance_stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "num_days": 0,
        }

    cum_ret = (1 + ret).cumprod()
    total_return = float(cum_ret.iloc[-1] - 1.0)

    avg_daily_ret = float(ret.mean())
    daily_vol = float(ret.std(ddof=1))

    annual_return = (
        (1 + avg_daily_ret) ** TRADING_DAYS_PER_YEAR - 1
        if avg_daily_ret != -1
        else -1
    )
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_vol > 0 else 0.0
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_drawdown = float(drawdown.min())

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "num_days": len(ret),
    }


def main(pair: str | None = None) -> Path:
    import argparse

    parser = argparse.ArgumentParser(description="Backtest pairs trading strategy")
    parser.add_argument("--pair", type=str, help="Pair like KO_PEP or XOM_CVX")
    args = parser.parse_args()

    if args.pair:
        pair = args.pair
    if not pair:
        raise ValueError("Must provide --pair like: KO_PEP")

    y_ticker, x_ticker = pair.split("_")

    in_path = PROCESSED_DIR / f"signals_{pair}.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Signals file not found: {in_path}. "
            "Run: python -m src.features.hedge_ratio and src.features.signals first."
        )

    df = pd.read_csv(in_path, parse_dates=["date"]).set_index("date")

    for col in [y_ticker, x_ticker, "position"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in {in_path}")

    beta = _estimate_beta(df[y_ticker], df[x_ticker])
    print(f"[backtest] using hedge ratio beta={beta:.4f}")

    ret = _compute_portfolio_returns(df, y_ticker, x_ticker, beta)

    df["strategy_return"] = ret
    df["equity_curve"] = (1 + ret.fillna(0)).cumprod()

    stats = _compute_performance_stats(ret)

    out_path = PROCESSED_DIR / f"backtest_results_{pair}.csv"
    df.to_csv(out_path)

    print("[backtest] ===== Performance summary =====")
    print(f"[backtest] Days: {stats['num_days']}")
    print(f"[backtest] Total return: {stats['total_return']*100:.2f}%")
    print(f"[backtest] Annual return: {stats['annual_return']*100:.2f}%")
    print(f"[backtest] Annual vol: {stats['annual_vol']*100:.2f}%")
    print(f"[backtest] Sharpe: {stats['sharpe']:.2f}")
    print(f"[backtest] Max drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f"[backtest] Saved time series to {out_path}")

    return out_path


if __name__ == "__main__":
    main()
