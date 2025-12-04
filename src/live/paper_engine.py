from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import argparse
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "adj_close_clean.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ENTRY_Z = 2.0
EXIT_Z = 0.5
TRADING_DAYS_PER_YEAR = 252
TRANSACTION_COST = 0.0002 

@dataclass
class Trade:
    date: pd.Timestamp
    leg: str       
    ticker: str
    qty: float
    price: float
    notional: float


class PaperBroker:
    def __init__(self, starting_cash: float = 100_000.0) -> None:
        self.cash: float = starting_cash
        self.positions: Dict[str, float] = {}   # ticker -> shares
        self.trades: List[Trade] = []
        self.fees_paid: float = 0.0 

    def ensure_ticker(self, ticker: str) -> None:
        if ticker not in self.positions:
            self.positions[ticker] = 0.0

    def trade(self, date: pd.Timestamp, ticker: str, qty: float, price: float, leg: str) -> None:
        self.ensure_ticker(ticker)
        notional = qty * price
        cost = abs(notional) * TRANSACTION_COST

        self.positions[ticker] += qty

        self.cash -= notional
        self.cash -= cost
        self.fees_paid += cost

        self.trades.append(
            Trade(
                date=date,
                leg=leg,
                ticker=ticker,
                qty=qty,
                price=price,
                notional=notional,
            )
        )


    def position(self, ticker: str) -> float:
        return self.positions.get(ticker, 0.0)

    def portfolio_value(self, prices: Dict[str, float]) -> float:
        eq = self.cash
        for ticker, qty in self.positions.items():
            if ticker in prices:
                eq += qty * prices[ticker]
        return eq


class PairsLiveEngine:
    def __init__(
        self,
        pair: str,
        starting_cash: float = 100_000.0,
        window: int = 60,
        entry_z: float = ENTRY_Z,
        exit_z: float = EXIT_Z,
        risk_frac: float = 0.5,
    ) -> None:
        self.pair = pair
        self.y_ticker, self.x_ticker = pair.split("_")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.risk_frac = risk_frac

        self.broker = PaperBroker(starting_cash=starting_cash)

        if not INTERIM_PATH.exists():
            raise FileNotFoundError(
                f"{INTERIM_PATH} not found. Run: python -m src.data.clean"
            )
        df = pd.read_csv(INTERIM_PATH, parse_dates=["date"]).set_index("date")
        for t in (self.y_ticker, self.x_ticker):
            if t not in df.columns:
                raise KeyError(f"{t} not found in {INTERIM_PATH}")
        self.prices_df = df[[self.y_ticker, self.x_ticker]].copy()

        hedge_path = PROCESSED_DIR / f"hedge_results_{pair}.csv"
        if not hedge_path.exists():
            raise FileNotFoundError(
                f"{hedge_path} not found. Run hedge_ratio for this pair first!"
            )
        hedge_df = pd.read_csv(hedge_path, parse_dates=["date"]).set_index("date")
        cols_to_drop = [self.y_ticker, self.x_ticker]
        hedge_df = hedge_df.drop(columns=cols_to_drop, errors="ignore")

        self.data = self.prices_df.join(hedge_df, how="inner")


        if "spread" not in self.data.columns:
            raise KeyError("spread column not found in hedge results.")

        z_cols = [c for c in self.data.columns if c.startswith("zscore_roll")]
        self.z_col = z_cols[0] if z_cols else None
        self.beta = float(self._infer_beta())


    def _infer_beta(self) -> float:
        y = self.data[self.y_ticker].astype(float)
        x = self.data[self.x_ticker].astype(float)
        df = pd.concat({"y": y, "x": x}, axis=1).dropna()

        X = sm.add_constant(df["x"])
        model = sm.OLS(df["y"], X).fit()
        beta = float(model.params["x"])
        return beta

    def _compute_zscore_series(self) -> pd.Series:
        if self.z_col is not None:
            return self.data[self.z_col].copy()

        spread = self.data["spread"]
        roll_mean = spread.rolling(self.window).mean()
        roll_std = spread.rolling(self.window).std(ddof=1)
        z = (spread - roll_mean) / roll_std
        z.name = f"zscore_roll_{self.window}"
        return z

    def run(self) -> pd.DataFrame:
        z = self._compute_zscore_series()
        equity_curve = []
        positions_y = []
        positions_x = []
        signals = []
        dates = self.data.index

        current_pos = 0 

        for i, date in enumerate(dates):
            row = self.data.loc[date]
            price_y = float(row[self.y_ticker])
            price_x = float(row[self.x_ticker])
            z_i = float(z.loc[date])

            signal = 0
            if z_i > self.entry_z:
                signal = -1  
            elif z_i < -self.entry_z:
                signal = 1  
            elif abs(z_i) < self.exit_z:
                signal = 0

            target_pos = current_pos
            if signal != 0:
                target_pos = signal
            elif abs(z_i) < self.exit_z:
                target_pos = 0

            if target_pos != current_pos:
                equity_now = self.broker.portfolio_value({self.y_ticker: price_y, self.x_ticker: price_x})
                base_notional = self.risk_frac * equity_now
                z_scale = min(2.0, abs(z_i) / max(self.entry_z, 1e-6))

                spread_notional = base_notional * z_scale

                unit = spread_notional / (price_y + abs(self.beta) * price_x)

                desired_y = target_pos * unit
                desired_x = -target_pos * self.beta * unit

                cur_y = self.broker.position(self.y_ticker)
                cur_x = self.broker.position(self.x_ticker)

                trade_y = desired_y - cur_y
                trade_x = desired_x - cur_x

                if abs(trade_y) > 1e-6:
                    self.broker.trade(date, self.y_ticker, trade_y, price_y, leg="Y")
                if abs(trade_x) > 1e-6:
                    self.broker.trade(date, self.x_ticker, trade_x, price_x, leg="X")

                current_pos = target_pos
            eq = self.broker.portfolio_value({self.y_ticker: price_y, self.x_ticker: price_x})
            equity_curve.append(eq)
            positions_y.append(self.broker.position(self.y_ticker))
            positions_x.append(self.broker.position(self.x_ticker))
            signals.append(current_pos)

        result = self.data[[self.y_ticker, self.x_ticker]].copy()
        result["zscore"] = z
        result["signal"] = signals
        result["pos_y"] = positions_y
        result["pos_x"] = positions_x
        result["equity"] = equity_curve

        result["ret"] = result["equity"].pct_change()
        stats = self._compute_stats(result["ret"])
        self._print_summary(stats)

        out_path = PROCESSED_DIR / f"paper_results_{self.pair}.csv"
        result.to_csv(out_path)
        print(f"[paper] saved paper trading run to {out_path} (rows={result.shape[0]})")

        return result

    def _compute_stats(self, ret: pd.Series) -> dict:
        ret = ret.dropna()
        if ret.empty:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "annual_vol": 0.0,
                "sharpe": 0.0,
                "num_days": 0,
                "max_drawdown": 0.0,
            }

        cum = (1 + ret).cumprod()
        total_return = float(cum.iloc[-1] - 1.0)
        avg_daily = float(ret.mean())
        vol_daily = float(ret.std(ddof=1))
        annual_return = (1 + avg_daily) ** TRADING_DAYS_PER_YEAR - 1 if avg_daily != -1 else -1
        annual_vol = vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR) if vol_daily > 0 else 0.0
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = float(dd.min())

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "num_days": len(ret),
            "max_drawdown": max_dd,
        }

    def _print_summary(self, s: dict) -> None:
        print("[paper] ===== Paper trading summary =====")
        print(f"[paper] Pair: {self.pair}, hedge beta≈{self.beta:.4f}")
        print(f"[paper] Days: {s['num_days']}")
        print(f"[paper] Total return: {s['total_return']*100:.2f}%")
        print(f"[paper] Annual return: {s['annual_return']*100:.2f}%")
        print(f"[paper] Annual vol: {s['annual_vol']*100:.2f}%")
        print(f"[paper] Sharpe: {s['sharpe']:.2f}")
        print(f"[paper] Max drawdown: {s['max_drawdown']*100:.2f}%")
        print(f"[paper] Total fees paid: ${self.broker.fees_paid:,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading engine for a pairs strategy")
    parser.add_argument("--pair", type=str, required=True, help="Pair like XOM_CVX or MA_V")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting cash")
    parser.add_argument("--risk-frac", type=float, default=0.5, help="Fraction of equity to deploy per spread (0-1)")
    parser.add_argument("--window", type=int, default=60, help="Rolling z-score window")
    parser.add_argument("--entry-z", type=float, default=ENTRY_Z)
    parser.add_argument("--exit-z", type=float, default=EXIT_Z)
    args = parser.parse_args()

    engine = PairsLiveEngine(pair=args.pair, starting_cash=args.cash, risk_frac=args.risk_frac, window=args.window, entry_z=args.entry_z, exit_z=args.exit_z,)
    engine.run()


if __name__ == "__main__":
    main()
