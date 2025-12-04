from pathlib import Path
import pandas as pd
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IN_PATH = PROJECT_ROOT / "data" / "processed" / "hedge_results.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "signals.csv"


ENTRY_Z = 2.0     
EXIT_Z = 0.5       


def _choose_z_column(df: pd.DataFrame) -> str:
    roll_cols = [c for c in df.columns if c.startswith("zscore_roll")]
    if roll_cols:
        return roll_cols[0]
    if "zscore_full" in df.columns:
        return "zscore_full"
    raise KeyError("No z-score column found in hedge_results.csv")


def _compute_signals(z: pd.Series) -> pd.DataFrame:
    signal = pd.Series(0, index=z.index, name="signal")

    signal[z > ENTRY_Z] = -1    
    signal[z < -ENTRY_Z] = 1   

    position = signal.copy().rename("position")

    for i in range(1, len(position)):
        if position.iat[i] == 0:
            position.iat[i] = position.iat[i - 1]

    flat_mask = z.abs() < EXIT_Z
    position[flat_mask] = 0

    return pd.concat([signal, position], axis=1)


def main(pair: str | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate trading signals for a pair")
    parser.add_argument("--pair", type=str, help="Pair like KO_PEP")
    args = parser.parse_args()

    if args.pair:
        pair = args.pair
    if not pair:
        raise ValueError("Must provide --pair like: KO_PEP")

    y_ticker, x_ticker = pair.split("_")

    in_path = OUT_DIR / f"hedge_results_{pair}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Input {in_path} not found. Run hedge_ratio first!")

    df = pd.read_csv(in_path, parse_dates=["date"]).set_index("date")

    z_col = _choose_z_column(df)
    z = df[z_col]

    sig_df = _compute_signals(z)

    df["signal"] = sig_df["signal"]
    df["position"] = sig_df["position"]

    out_path = OUT_DIR / f"signals_{pair}.csv"
    df.to_csv(out_path)

    print("[signals] ===== Signal generation =====")
    print(f"[signals] pair: {pair}")
    print(f"[signals] using z column: {z_col}")
    print(f"[signals] entry_z={ENTRY_Z}, exit_z={EXIT_Z}")
    print(f"[signals] saved {out_path} (rows={df.shape[0]})")

    return out_path


if __name__ == "__main__":
    main()
