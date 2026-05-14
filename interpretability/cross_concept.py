import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

warnings.simplefilter("ignore", category=ConvergenceWarning)

from utils.autointerp import encode_through_sae
from utils.data import (
    MODEL_NAME,
    get_datasets,
    get_default_sae_layer,
    get_xy_traintest,
)
from utils.sae import layer_to_sae_ids_gemma3, sae_id_to_sae_gemma3

TAU = 0.1
LOGREG_TOP_K = 100
SAMPLES_PER_CONCEPT = 64
NUM_TRAIN = 1024


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    # compute CH, neg-DB, silhouette, and MPCD-ratio for a labelled embedding
    unique, counts = np.unique(labels, return_counts=True)
    nan = float("nan")
    if len(unique) < 2 or np.any(counts < 2):
        return {"ch": nan, "neg_db": nan, "silhouette": nan, "mpcd_ratio": nan}

    ch = float(calinski_harabasz_score(X, labels))

    try:
        neg_db = -float(davies_bouldin_score(X, labels))
    except Exception:
        neg_db = nan

    try:
        sil = float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        sil = nan

    centroids = np.stack([X[labels == c].mean(axis=0) for c in unique])
    n_c = len(unique)
    dists = [
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(n_c) for j in range(i + 1, n_c)
    ]
    mean_centroid_dist = float(np.mean(dists)) if dists else nan

    intra = [float(np.std(X[labels == c], axis=0).mean()) for c in unique if (labels == c).sum() >= 2]
    mean_intra_std = float(np.mean(intra)) if intra else nan

    mpcd_ratio = (mean_centroid_dist / mean_intra_std) if mean_intra_std > 1e-10 else nan

    return {"ch": ch, "neg_db": neg_db, "silhouette": sil, "mpcd_ratio": mpcd_ratio}


def pairwise_separation(X: np.ndarray, labels: np.ndarray,
                        concept_names: list[str]) -> pd.DataFrame:
    # compute centroid distances and pooled-std separation ratios for all concept pairs
    unique = np.unique(labels)
    rows = []
    for i, ci in enumerate(unique):
        for j, cj in enumerate(unique):
            if j <= i:
                continue
            Xi, Xj = X[labels == ci], X[labels == cj]
            dist = float(np.linalg.norm(Xi.mean(0) - Xj.mean(0)))
            pooled_std = (np.std(Xi, axis=0).mean() + np.std(Xj, axis=0).mean()) / 2 + 1e-8
            rows.append({
                "concept_i": concept_names[ci] if ci < len(concept_names) else str(ci),
                "concept_j": concept_names[cj] if cj < len(concept_names) else str(cj),
                "centroid_dist": dist,
                "pooled_std": float(pooled_std),
                "separation_ratio": dist / float(pooled_std),
            })
    return pd.DataFrame(rows)


def multiclass_top_k_dims(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    # fit a one-vs-rest logistic regression and return the top-k dims by mean |coef|
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", multi_class="ovr")
    clf.fit(X, y)
    return np.argsort(np.abs(clf.coef_).mean(axis=0))[::-1][:k]


def encode_pos_through_sae(sae, X_pos_raw: np.ndarray, device: str) -> np.ndarray:
    X_t = torch.from_numpy(X_pos_raw).float()
    Z = encode_through_sae(sae, X_t, device)
    return Z


def _delta(a: float, b: float) -> float | None:
    if np.isnan(a) or np.isnan(b):
        return None
    return float(a - b)


def _json_default(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    return float(x)


def _print_metrics(tag: str, m: dict):
    print(f"  {tag:<28} ch={m['ch']:>9.2f}  neg_db={m['neg_db']:>8.4f}  "
          f"sil={m['silhouette']:>7.4f}  mpcd={m['mpcd_ratio']:>8.3f}")


def _print_row(tag: str, m: dict):
    print(f"{tag:<30} {m['ch']:>10.2f} {m['neg_db']:>10.4f} "
          f"{m['silhouette']:>12.4f} {m['mpcd_ratio']:>10.3f}")


if __name__ == "__main__":
    DEVICE = "cuda:0"
    LAYER = None
    MAX_DATASETS = None
    LOGREG_TOP_K = LOGREG_TOP_K
    TAU = TAU
    SAMPLES_PER_CONCEPT = SAMPLES_PER_CONCEPT

    layer = LAYER if LAYER is not None else get_default_sae_layer()
    datasets = get_datasets()
    if MAX_DATASETS is not None:
        datasets = datasets[:MAX_DATASETS]

    sae_ids = layer_to_sae_ids_gemma3(layer)
    print(f"Layer {layer} | SAE types: {len(sae_ids)} | device {DEVICE}")
    for sid in sae_ids:
        print(f"  {sid}")
    print(f"Concepts: {len(datasets)} | samples_per_concept={SAMPLES_PER_CONCEPT} "
          f"| logreg_top_k={LOGREG_TOP_K} | tau={TAU}\n")

    out_dir = Path(f"results/cross_concept_sep_{MODEL_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 1: collecting positive-class activations...")
    pos_chunks: list[np.ndarray] = []
    labels_list: list[int] = []
    concept_names: list[str] = []
    concept_idx = 0

    for dataset in datasets:
        try:
            X_tr, y_tr, _, _ = get_xy_traintest(NUM_TRAIN, dataset, layer, seed=42)
        except Exception as e:
            print(f"  [skip] {dataset}: {e}")
            continue

        if len(np.unique(y_tr)) < 2:
            continue

        X_tr_np = X_tr.numpy() if hasattr(X_tr, "numpy") else np.asarray(X_tr, dtype=np.float32)
        del X_tr

        pos_idx = np.where(y_tr == 1)[0]
        n_take = min(SAMPLES_PER_CONCEPT, len(pos_idx))
        if n_take < 2:
            continue

        rng = np.random.default_rng(42 + concept_idx)
        chosen = rng.choice(pos_idx, n_take, replace=False)
        pos_chunks.append(X_tr_np[chosen].astype(np.float32))
        del X_tr_np

        labels_list.extend([concept_idx] * n_take)
        concept_names.append(dataset)
        print(f"  [{concept_idx+1}] {dataset}: {n_take} examples")
        concept_idx += 1

    if concept_idx < 3:
        print("Need 3+ concepts. Exiting.")
    else:
        labels = np.array(labels_list, dtype=np.int32)
        X_pos_all = np.vstack(pos_chunks)
        del pos_chunks
        print(f"\nCollected {X_pos_all.shape[0]} examples across {concept_idx} concepts "
              f"({X_pos_all.nbytes / 1e6:.1f} MB raw)\n")

        print(f"Phase 2: fitting multiclass logreg -> top-{LOGREG_TOP_K} dims...")
        top_k_idx = multiclass_top_k_dims(X_pos_all, labels, LOGREG_TOP_K)
        X_lr_all = X_pos_all[:, top_k_idx].copy()

        print("Phase 3: raw / logreg cluster metrics...")
        metrics_raw = cluster_metrics(X_pos_all, labels)
        metrics_lr  = cluster_metrics(X_lr_all, labels)
        df_pw_raw = pairwise_separation(X_pos_all, labels, concept_names)
        df_pw_lr  = pairwise_separation(X_lr_all,  labels, concept_names)

        _print_metrics("raw",    metrics_raw)
        _print_metrics("logreg", metrics_lr)

        baseline = {
            "metrics_raw": metrics_raw,
            "metrics_logreg": metrics_lr,
            "mean_pairwise_sep_ratio_raw":    float(df_pw_raw["separation_ratio"].mean()),
            "mean_pairwise_sep_ratio_logreg": float(df_pw_lr["separation_ratio"].mean()),
        }

        df_pw_raw["rep"] = "raw"
        df_pw_lr["rep"]  = "logreg"

        all_sae_results: list[dict] = []
        pairwise_frames: list[pd.DataFrame] = [df_pw_raw, df_pw_lr]

        for sae_id in sae_ids:
            sae_type = sae_id.split("/")[-1]
            print(f"\nPhase 4 [{sae_type}]: loading SAE...")

            try:
                sae = sae_id_to_sae_gemma3(sae_id, DEVICE)
            except Exception as e:
                print(f"  [skip] {sae_id}: load failed - {e}")
                continue

            try:
                Z = encode_pos_through_sae(sae, X_pos_all, DEVICE)
            except Exception as e:
                print(f"  [skip] {sae_id}: encode failed - {e}")
                del sae; gc.collect()
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue

            del sae; gc.collect()
            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()

            print(f"  SAE latents: {Z.shape} ({Z.nbytes / 1e6:.1f} MB)")

            fire_frac = (Z > 0).mean(axis=0)
            mask = fire_frac > TAU
            if mask.sum() < 2:
                mask = fire_frac > 0.01
            Z_f = Z[:, mask].copy()
            del Z; gc.collect()
            print(f"  After tau filter: {mask.sum()} latents kept")

            metrics_sae = cluster_metrics(Z_f, labels)
            df_pw_sae = pairwise_separation(Z_f, labels, concept_names)
            del Z_f; gc.collect()

            _print_metrics(sae_type, metrics_sae)
            df_pw_sae["rep"] = sae_type
            pairwise_frames.append(df_pw_sae)

            sae_result = {
                "sae_id": sae_id, "sae_type": sae_type,
                "n_latents_after_tau": int(mask.sum()),
                "metrics_sae": metrics_sae,
                "deltas_vs_logreg": {m: _delta(metrics_sae[m], metrics_lr[m])
                                     for m in ("ch", "neg_db", "silhouette", "mpcd_ratio")},
                "deltas_vs_raw":    {m: _delta(metrics_sae[m], metrics_raw[m])
                                     for m in ("ch", "neg_db", "silhouette", "mpcd_ratio")},
                "mean_pairwise_sep_ratio": float(df_pw_sae["separation_ratio"].mean()),
            }
            all_sae_results.append(sae_result)

            with open(out_dir / f"results_{sae_type}.json", "w") as f:
                json.dump({**baseline, **sae_result,
                           "layer": layer, "tau": TAU, "logreg_top_k": LOGREG_TOP_K,
                           "concept_names": concept_names, "n_concepts": concept_idx,
                           "samples_per_concept": SAMPLES_PER_CONCEPT},
                          f, indent=2, default=_json_default)

        df_pairwise = pd.concat(pairwise_frames, ignore_index=True)
        pw_path = out_dir / "pairwise_distances.csv"
        df_pairwise.to_csv(pw_path, index=False)
        print(f"\nSaved pairwise distances -> {pw_path}")

        summary = {
            "layer": layer, "n_concepts": concept_idx,
            "samples_per_concept": SAMPLES_PER_CONCEPT,
            "logreg_top_k": LOGREG_TOP_K, "tau": TAU,
            "concept_names": concept_names,
            **baseline, "sae_results": all_sae_results,
        }
        json_path = out_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        print(f"Saved summary -> {json_path}")

        print(f"\n{'Rep':<30} {'CH':>10} {'neg_DB':>10} {'Silhouette':>12} {'MPCD':>10}")
        print("-" * 76)
        _print_row("raw",    metrics_raw)
        _print_row("logreg", metrics_lr)
        for r in all_sae_results:
            _print_row(r["sae_type"], r["metrics_sae"])
