import glob
import os
import pickle

import pandas as pd
from tqdm import tqdm

from utils.data import (
    MODEL_NAME,
    get_OOD_datasets,
    get_class_imbalance,
    get_default_sae_layer,
    get_layers,
    get_numbered_binary_tags,
    get_training_sizes,
)

BASELINE_METHODS = ["logreg"]
METRIC = "test_auc"


def _load_pkl_setting(setting_dir: str) -> pd.DataFrame:
    # load all SAE probe pickle files for a given setting into a single DataFrame
    pattern = f"data/sae_probes_{MODEL_NAME}/{setting_dir}/*.pkl"
    files = glob.glob(pattern)
    if not files:
        print(f"  No SAE probe files found at: {pattern}")
        return pd.DataFrame()

    all_rows: list[dict] = []
    for path in tqdm(files, desc=f"Loading {setting_dir}", leave=False):
        try:
            with open(path, "rb") as f:
                metrics_list = pickle.load(f)
        except Exception as e:
            print(f"  Bad pkl {path}: {e}")
            continue
        if metrics_list:
            all_rows.extend(metrics_list)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def _read_baseline_csv(path: str) -> float | None:
    # read the test AUC from a single baseline result CSV
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or METRIC not in df.columns:
        return None
    return float(df[METRIC].iloc[0])


def _read_ood_baseline_csv(path: str) -> float | None:
    # read OOD baseline AUC, checking both column name variants
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    if "test_auc_baseline" in df.columns:
        return float(df["test_auc_baseline"].iloc[0])
    if METRIC in df.columns:
        return float(df[METRIC].iloc[0])
    return None


def _norm_frac(x) -> float | None:
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def _read_imbalance_baseline_csv(layer: int | str, dataset: str, method: str, frac: float) -> float | None:
    # tolerates frac naming variants like 0.1 vs 0.10
    base = f"data/baseline_results_{MODEL_NAME}/imbalance/allruns"
    frac_candidates = [f"{frac:.2f}", f"{frac:g}", str(frac)]
    seen: set[str] = set()
    for frac_str in frac_candidates:
        if frac_str in seen:
            continue
        seen.add(frac_str)
        path = f"{base}/layer{layer}_{dataset}_{method}_frac{frac_str}.csv"
        val = _read_baseline_csv(path)
        if val is not None:
            return val
    return None


def _build_sae_lookup(sae_df: pd.DataFrame, group_cols: list[str]) -> dict[tuple, dict]:
    # build a dict mapping group keys to the best-AUC SAE probe row for that group
    if sae_df.empty:
        return {}
    if METRIC not in sae_df.columns:
        raise KeyError(
            f"Column '{METRIC}' not found in SAE probe data. "
            f"Available: {list(sae_df.columns)}"
        )
    idx = sae_df.groupby(group_cols)[METRIC].idxmax()
    cols = group_cols + [METRIC, "k", "sae_id"]
    best_df = sae_df.loc[idx.tolist(), cols]
    lookup: dict[tuple, dict] = {}
    for _, row in best_df.iterrows():
        key = tuple(row[c] for c in group_cols)
        lookup[key] = {
            "sae_best_test_auc": float(row[METRIC]),
            "sae_best_k":        row["k"],
            "sae_best_sae_id":   row["sae_id"],
        }
    return lookup


def _attach_sae(row: dict, entry: dict | None, baseline_auc: float | None) -> dict:
    if entry is not None:
        sae_auc = entry["sae_best_test_auc"]
        row["sae_best_test_auc"] = sae_auc
        row["sae_best_k"]        = entry["sae_best_k"]
        row["sae_best_sae_id"]   = entry["sae_best_sae_id"]
        row["sae_better"] = (sae_auc > baseline_auc) if baseline_auc is not None else None
    else:
        row["sae_best_test_auc"] = None
        row["sae_best_k"]        = None
        row["sae_best_sae_id"]   = None
        row["sae_better"]        = None
    return row


def _print_summary(out_df: pd.DataFrame):
    valid = out_df.dropna(subset=["sae_better"])
    if valid.empty:
        print("  (no rows with both values present — cannot compute summary)")
        return
    n_better = int(valid["sae_better"].sum())
    print(f"  SAE better than baseline : {n_better} / {len(valid)} ({100*n_better/len(valid):.1f}%)")
    print("  By method:")
    for method, grp in valid.groupby("method"):
        n = int(grp["sae_better"].sum())
        print(f"    {method:10s}: {n}/{len(grp)} ({100*n/len(grp):.1f}%)")


def _write(out_df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"  Wrote {len(out_df)} rows → {out_path}")


def build_comparison_normal():
    # compare SAE probes vs baseline classifiers for the normal setting across all layers
    print("\n=== normal ===")
    layers   = get_layers(MODEL_NAME)
    datasets = get_numbered_binary_tags()
    print(f"  Layers: {layers}  |  Datasets: {len(datasets)}")

    sae_df = _load_pkl_setting("normal_setting")
    if not sae_df.empty:
        print(f"  Loaded {len(sae_df)} SAE probe rows.")
    sae_lookup = _build_sae_lookup(sae_df, ["dataset", "layer"])

    rows: list[dict] = []
    for layer in layers:
        for dataset in datasets:
            sae_entry = sae_lookup.get((dataset, layer))
            for method in BASELINE_METHODS:
                p = (
                    f"data/baseline_results_{MODEL_NAME}/normal/allruns/"
                    f"layer{layer}_{dataset}_{method}.csv"
                )
                baseline_auc = _read_baseline_csv(p)
                if sae_entry is None or baseline_auc is None:
                    continue
                row: dict = {"dataset": dataset, "layer": layer, "method": method,
                             "baseline_test_auc": baseline_auc}
                rows.append(_attach_sae(row, sae_entry, baseline_auc))

    out_df = pd.DataFrame(rows)
    _write(out_df, f"results/comparison_{MODEL_NAME}/normal_sae_vs_baselines.csv")
    _print_summary(out_df)
    return out_df


def build_comparison_scarcity():
    # compare SAE probes vs baselines across different training set sizes
    print("\n=== scarcity ===")
    layer    = get_default_sae_layer(MODEL_NAME)
    datasets = get_numbered_binary_tags()
    sizes    = get_training_sizes()
    print(f"  Layer: {layer}  |  Datasets: {len(datasets)}  |  Train sizes: {list(sizes)}")

    sae_df = _load_pkl_setting("scarcity_setting")
    if not sae_df.empty:
        print(f"  Loaded {len(sae_df)} SAE probe rows.")
    sae_lookup = _build_sae_lookup(sae_df, ["dataset", "layer", "num_train"])

    rows: list[dict] = []
    for num_train in sizes:
        for dataset in datasets:
            sae_entry = sae_lookup.get((dataset, layer, num_train))
            for method in BASELINE_METHODS:
                p = (
                    f"data/baseline_results_{MODEL_NAME}/scarcity/allruns/"
                    f"layer{layer}_{dataset}_{method}_numtrain{num_train}.csv"
                )
                baseline_auc = _read_baseline_csv(p)
                if sae_entry is None or baseline_auc is None:
                    continue
                row: dict = {"dataset": dataset, "layer": layer, "num_train": num_train,
                             "method": method, "baseline_test_auc": baseline_auc}
                rows.append(_attach_sae(row, sae_entry, baseline_auc))

    out_df = pd.DataFrame(rows)
    _write(out_df, f"results/comparison_{MODEL_NAME}/scarcity_sae_vs_baselines.csv")
    _print_summary(out_df)
    return out_df


def build_comparison_imbalance():
    # compare SAE probes vs baselines across different positive-class fractions
    print("\n=== imbalance ===")
    layer    = get_default_sae_layer(MODEL_NAME)
    datasets = get_numbered_binary_tags()
    frac_vals: list[float] = []
    for f in get_class_imbalance():
        frac = _norm_frac(f)
        if frac is not None:
            frac_vals.append(frac)
    fracs = sorted(set(frac_vals))
    print(f"  Layer: {layer}  |  Datasets: {len(datasets)}  |  Fracs: {len(fracs)}")

    sae_df = _load_pkl_setting("imbalance_setting")
    if not sae_df.empty:
        print(f"  Loaded {len(sae_df)} SAE probe rows.")
    if not sae_df.empty:
        if "frac" not in sae_df.columns:
            raise KeyError(
                f"Column 'frac' not found in SAE probe data for imbalance setting. "
                f"Available: {list(sae_df.columns)}"
            )
        sae_df["frac"] = pd.to_numeric(sae_df["frac"], errors="coerce").round(2)
        sae_df = sae_df.dropna(subset=["frac"])
    sae_lookup = _build_sae_lookup(sae_df, ["dataset", "layer", "frac"])

    rows: list[dict] = []
    for frac in fracs:
        for dataset in datasets:
            sae_entry = sae_lookup.get((dataset, layer, frac))
            for method in BASELINE_METHODS:
                baseline_auc = _read_imbalance_baseline_csv(layer, dataset, method, frac)
                if sae_entry is None or baseline_auc is None:
                    continue
                row: dict = {"dataset": dataset, "layer": layer, "frac": frac,
                             "method": method, "baseline_test_auc": baseline_auc}
                rows.append(_attach_sae(row, sae_entry, baseline_auc))

    out_df = pd.DataFrame(rows)
    _write(out_df, f"results/comparison_{MODEL_NAME}/imbalance_sae_vs_baselines.csv")
    _print_summary(out_df)
    return out_df


def build_comparison_ood():
    # compare SAE probes vs baselines on out-of-distribution test sets
    print("\n=== OOD ===")
    layer = get_default_sae_layer(MODEL_NAME)
    datasets = get_OOD_datasets()
    print(f"  Layer: {layer}  |  OOD datasets: {len(datasets)}")

    sae_df = _load_pkl_setting("OOD_setting")
    if not sae_df.empty:
        print(f"  Loaded {len(sae_df)} SAE probe rows.")
    sae_lookup = _build_sae_lookup(sae_df, ["dataset", "layer"])

    rows: list[dict] = []
    for dataset in datasets:
        sae_entry = sae_lookup.get((dataset, layer))
        baseline_path = f"data/baseline_results_{MODEL_NAME}/ood/allruns/{dataset}.csv"
        baseline_auc = _read_ood_baseline_csv(baseline_path)
        if sae_entry is None or baseline_auc is None:
            continue
        row: dict = {
            "dataset": dataset,
            "layer": layer,
            "method": "logreg",
            "baseline_test_auc": baseline_auc,
        }
        rows.append(_attach_sae(row, sae_entry, baseline_auc))

    out_df = pd.DataFrame(rows)
    _write(out_df, f"results/comparison_{MODEL_NAME}/ood_sae_vs_baselines.csv")
    _print_summary(out_df)
    return out_df


if __name__ == "__main__":
    SETTING = None   # one of: normal, scarcity, imbalance, OOD — or None to run all

    if SETTING in (None, "normal"):
        build_comparison_normal()
    if SETTING in (None, "scarcity"):
        build_comparison_scarcity()
    if SETTING in (None, "imbalance"):
        build_comparison_imbalance()
    if SETTING in (None, "OOD"):
        build_comparison_ood()
