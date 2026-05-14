import pandas as pd
from utils.data import MODEL_NAME


def _aggregate_setting(setting_name: str):
    # count per-dataset win rates for SAE vs baseline and write a ranked CSV
    print(f"\n=== {setting_name.capitalize()} Aggregation ===")
    in_path = f"results/comparison_{MODEL_NAME}/{setting_name}_sae_vs_baselines.csv"
    out_path = f"results/comparison_{MODEL_NAME}/{setting_name}_wins_by_dataset.csv"

    try:
        df = pd.read_csv(in_path)
    except FileNotFoundError:
        print(f"  Missing input: {in_path}")
        return pd.DataFrame()

    agg = []
    for dataset, grp in df.groupby("dataset"):
        valid = grp.dropna(subset=["sae_better"])
        if valid.empty:
            continue
        sae_wins = int((valid["sae_better"] == True).sum())
        baseline_wins = int((valid["sae_better"] == False).sum())
        total = len(valid)
        win_rate = 100 * sae_wins / total if total > 0 else 0
        agg.append({
            "dataset": dataset,
            "sae_wins": sae_wins,
            "baseline_wins": baseline_wins,
            "total": total,
            "sae_win_rate": f"{win_rate:.1f}%",
        })

    out_df = pd.DataFrame(agg)
    if out_df.empty:
        print("  No rows with non-null sae_better; nothing to write")
        return out_df

    out_df = out_df.sort_values("sae_win_rate", ascending=False)
    out_df.to_csv(out_path, index=False)
    print(f"  Wrote {len(out_df)} rows -> {out_path}")
    print(f"  Sample:\n{out_df.head()}")
    return out_df


def aggregate_scarcity():
    return _aggregate_setting("scarcity")


def aggregate_imbalance():
    return _aggregate_setting("imbalance")


def aggregate_ood():
    return _aggregate_setting("ood")


if __name__ == "__main__":
    aggregate_scarcity()
    aggregate_imbalance()
    aggregate_ood()
    print("\nDone")
