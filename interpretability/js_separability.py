import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.data import (
    MODEL_NAME, get_datasets, get_default_sae_layer,
    get_xvals, get_yvals, get_train_test_indices,
)
from utils.sae import layer_to_sae_ids_gemma3, sae_id_to_sae_gemma3
from utils.autointerp import encode_through_sae

LOGREG_TOP_K = 100

warnings.filterwarnings("ignore", category=UserWarning)

TOP_T   = 80
N_BINS  = 2048
TAU     = 0.1
MAX_TRAIN = 1024
MAX_TOTAL = 1500


def salient_neuron_overlap(X: np.ndarray, y: np.ndarray, top_t: int = TOP_T) -> float:
    # Jaccard IoU of the top-T mean-activation neurons per class; lower = more class-specific
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("salient_neuron_overlap requires exactly 2 classes")

    t = min(top_t, X.shape[1])
    sets = []
    for c in classes:
        means = X[y == c].mean(axis=0)
        top_idx = set(np.argsort(means)[-t:].tolist())
        sets.append(top_idx)

    intersection = sets[0] & sets[1]
    union        = sets[0] | sets[1]
    return len(intersection) / len(union) if union else 1.0


def _entropy_from_hist(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def js_separability_score(
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = N_BINS,
    tau: float = TAU,
    is_sae: bool = False,
) -> float:
    # mean sqrt(JSD) across all (active) neurons; higher = better class separation
    classes = np.unique(y)
    k = len(classes)
    if k != 2:
        raise ValueError("js_separability_score requires exactly 2 classes")

    _, d = X.shape
    scores = []

    x_min = X.min(axis=0)
    x_max = X.max(axis=0)

    class_masks = [y == c for c in classes]

    for j in range(d):
        col = X[:, j]

        if is_sae and (col != 0).mean() < tau:
            continue

        lo, hi = float(x_min[j]), float(x_max[j])
        if lo == hi:
            scores.append(0.0)
            continue

        edges = np.linspace(lo, hi, n_bins + 1)

        class_counts = []
        for mask in class_masks:
            counts, _ = np.histogram(col[mask], bins=edges)
            counts = counts + 1e-10
            class_counts.append(counts)

        weights = np.full(k, 1.0 / k)
        mixture = np.zeros_like(class_counts[0], dtype=float)
        for w, h in zip(weights, class_counts):
            mixture += w * h

        h_mixture = _entropy_from_hist(mixture)
        h_classes  = [_entropy_from_hist(h) for h in class_counts]
        h_mean     = float(np.mean(h_classes))

        jsd = max(0.0, h_mixture - h_mean)
        d_js = np.sqrt(jsd)
        scores.append(d_js)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def _fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def _apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def train_logreg_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    # cross-validate C on a held-out 20% split and refit on the full training set
    C_grid = np.logspace(5, -5, 10)

    X_subtr, X_val, y_subtr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42,
    )

    best_C = 1.0
    best_auc = -1.0

    for C in C_grid:
        mean_sub, std_sub = _fit_standardizer(X_subtr)
        X_subtr_std = _apply_standardizer(X_subtr, mean_sub, std_sub)
        X_val_std = _apply_standardizer(X_val, mean_sub, std_sub)

        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_subtr_std, y_subtr)

        try:
            val_proba = clf.predict_proba(X_val_std)[:, 1]
            auc = float(roc_auc_score(y_val, val_proba))
        except Exception:
            continue

        if auc > best_auc:
            best_auc = auc
            best_C = C

    mean, std = _fit_standardizer(X_train)
    X_train_std = _apply_standardizer(X_train, mean, std)
    best_clf = LogisticRegression(C=best_C, max_iter=1000, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_clf.fit(X_train_std, y_train)

    return best_clf, mean, std


def top_k_logreg_dims(clf: LogisticRegression, k: int) -> np.ndarray:
    weights = np.abs(clf.coef_[0])
    return np.argsort(weights)[-k:]


def compute_dataset_metrics(
    dataset: str,
    layer: int,
    sae,
    device: str,
    logreg_top_k: int = LOGREG_TOP_K,
) -> dict | None:
    # compute SNO and JSS for SAE, logistic-regression top dims, and raw activations
    try:
        X_raw = get_xvals(dataset, layer, MODEL_NAME).float().numpy()
        y_all_raw = np.asarray(get_yvals(dataset))
    except FileNotFoundError:
        return None

    N = min(len(X_raw), MAX_TOTAL)
    if N < 120:
        return None

    X_raw = X_raw[:N]
    y_all_raw = y_all_raw[:N]

    num_train = min(MAX_TRAIN, N - 100)
    num_test = N - num_train - 1
    if num_train < 10 or num_test < 10:
        return None

    try:
        train_idx, test_idx = get_train_test_indices(
            y_all_raw, num_train, num_test, pos_ratio=0.5, seed=42
        )
    except ValueError:
        return None

    X_train_np = X_raw[train_idx]
    X_test_np  = X_raw[test_idx]
    y_train_np = y_all_raw[train_idx].astype(int)
    y_test_np  = y_all_raw[test_idx].astype(int)

    X_all = np.concatenate([X_train_np, X_test_np], axis=0)
    y_all = np.concatenate([y_train_np, y_test_np], axis=0)

    X_sae_all   = encode_through_sae(sae, X_all,      device)
    X_sae_train = encode_through_sae(sae, X_train_np, device)
    X_sae_test  = encode_through_sae(sae, X_test_np,  device)

    sno_sae = salient_neuron_overlap(X_sae_all, y_all, top_t=TOP_T)
    jss_sae = js_separability_score( X_sae_all, y_all, is_sae=True)

    clf, _, _ = train_logreg_auc(X_train_np, y_train_np)
    k   = min(logreg_top_k, X_train_np.shape[1])
    top_dims = top_k_logreg_dims(clf, k)

    X_lr_all   = X_all[:, top_dims]
    X_lr_train = X_train_np[:, top_dims]
    X_lr_test  = X_test_np[:, top_dims]

    sno_logreg = salient_neuron_overlap(X_lr_all, y_all, top_t=min(TOP_T, k))
    jss_logreg = js_separability_score( X_lr_all, y_all, is_sae=False)

    sno_raw = salient_neuron_overlap(X_all, y_all, top_t=TOP_T)
    jss_raw = js_separability_score( X_all, y_all, is_sae=False)

    def _auc(Xtr, Xte, ytr, yte):
        clf2, mean2, std2 = train_logreg_auc(Xtr, ytr)
        Xte_std = _apply_standardizer(Xte, mean2, std2)
        proba = clf2.predict_proba(Xte_std)[:, 1]
        try:
            return float(roc_auc_score(yte, proba))
        except Exception:
            return float("nan")

    auc_sae    = _auc(X_sae_train, X_sae_test, y_train_np, y_test_np)
    auc_logreg = _auc(X_lr_train,  X_lr_test,  y_train_np, y_test_np)
    auc_raw    = _auc(X_train_np,  X_test_np,  y_train_np, y_test_np)

    return {
        "dataset": dataset,
        "layer":   layer,
        "sno_sae":    sno_sae,
        "sno_logreg": sno_logreg,
        "sno_raw":    sno_raw,
        "jss_sae":    jss_sae,
        "jss_logreg": jss_logreg,
        "jss_raw":    jss_raw,
        "auc_sae":    auc_sae,
        "auc_logreg": auc_logreg,
        "auc_raw":    auc_raw,
        "sno_sae_vs_logreg": sno_logreg - sno_sae,
        "jss_sae_vs_logreg": jss_sae   - jss_logreg,
    }


if __name__ == "__main__":
    DEVICE = "cuda:0"
    LAYER = None
    SAE_ID = None
    MAX_DATASETS = None
    LOGREG_TOP_K = LOGREG_TOP_K

    layer = LAYER or get_default_sae_layer()
    sae_ids = layer_to_sae_ids_gemma3(layer)
    sae_id  = SAE_ID or sae_ids[0]

    print(f"Model  : {MODEL_NAME}")
    print(f"Layer  : {layer}")
    print(f"SAE ID : {sae_id}")
    print(f"Device : {DEVICE}")

    print("Loading SAE ...")
    sae = sae_id_to_sae_gemma3(sae_id, device=DEVICE)

    datasets = get_datasets()
    if MAX_DATASETS:
        datasets = datasets[:MAX_DATASETS]
    print(f"Datasets: {len(datasets)}")

    results_dir = f"results/js_separability_{MODEL_NAME}"
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for dataset in tqdm(datasets, desc="Datasets"):
        row = compute_dataset_metrics(dataset, layer, sae, DEVICE, logreg_top_k=LOGREG_TOP_K)
        if row is None:
            tqdm.write(f"  Skipped {dataset} (missing data)")
            continue
        rows.append(row)
        tqdm.write(
            f"  {dataset[:38]:38s} | "
            f"SNO sae={row['sno_sae']:.3f} lr={row['sno_logreg']:.3f} "
            f"d={row['sno_sae_vs_logreg']:+.3f} | "
            f"JSS sae={row['jss_sae']:.4f} lr={row['jss_logreg']:.4f} "
            f"d={row['jss_sae_vs_logreg']:+.4f}"
        )

    if not rows:
        print("No results computed — check that activation files are present.")
    else:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(results_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(df)} rows -> {csv_path}")

        summary = {
            "model": MODEL_NAME, "layer": layer, "sae_id": sae_id,
            "n_datasets": len(df),
            "params": {"top_t": TOP_T, "n_bins": N_BINS, "tau": TAU, "logreg_top_k": LOGREG_TOP_K},
            "salient_neuron_overlap": {
                "mean_sae":           float(df["sno_sae"].mean()),
                "mean_logreg":        float(df["sno_logreg"].mean()),
                "mean_raw":           float(df["sno_raw"].mean()),
                "mean_sae_vs_logreg": float(df["sno_sae_vs_logreg"].mean()),
                "frac_sae_wins":      float((df["sno_sae_vs_logreg"] > 0).mean()),
            },
            "js_separability": {
                "mean_sae":           float(df["jss_sae"].mean()),
                "mean_logreg":        float(df["jss_logreg"].mean()),
                "mean_raw":           float(df["jss_raw"].mean()),
                "mean_sae_vs_logreg": float(df["jss_sae_vs_logreg"].mean()),
                "frac_sae_wins":      float((df["jss_sae_vs_logreg"] > 0).mean()),
            },
            "probe_auc": {
                "mean_auc_sae":    float(df["auc_sae"].mean()),
                "mean_auc_logreg": float(df["auc_logreg"].mean()),
                "mean_auc_raw":    float(df["auc_raw"].mean()),
            },
        }

        json_path = os.path.join(results_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary -> {json_path}")

        sno = summary["salient_neuron_overlap"]
        jss = summary["js_separability"]
        print(f"\nSalient Neuron Overlap: SAE={sno['mean_sae']:.4f}  LogReg={sno['mean_logreg']:.4f}  Raw={sno['mean_raw']:.4f}")
        print(f"JS Separability:        SAE={jss['mean_sae']:.6f}  LogReg={jss['mean_logreg']:.6f}  Raw={jss['mean_raw']:.6f}")
