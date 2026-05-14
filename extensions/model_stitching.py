import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from torchvision.models import ResNet50_Weights, resnet50

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
NUM_TRAIN_TEXT = 1024
RIDGE_ALPHA = 1.0
MAX_CONCEPTS = 100
CIFAR_ROOT = "./data/cifar"


class ResNetFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        m = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)   # (N, 2048)


def extract_cifar100_features(
    n_classes: int,
    samples_per_class: int,
    root: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    # extract ResNet50 penultimate features for CIFAR-100 classes
    S_tr = samples_per_class // 2
    S_te = samples_per_class - S_tr

    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = torchvision.datasets.CIFAR100(root=root, train=True,
                                              download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR100(root=root, train=False,
                                              download=True, transform=transform)
    class_names = train_ds.classes[:n_classes]

    model = ResNetFeatures()
    model.eval()

    def collect(ds, n_per_class, n_classes):
        targets = np.array(ds.targets)
        chunks_x, chunks_y = [], []
        for c in range(n_classes):
            idx = np.where(targets == c)[0]
            rng = np.random.default_rng(c)
            chosen = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
            imgs = torch.stack([ds[i][0] for i in chosen])
            with torch.no_grad():
                feats = model(imgs).numpy()
            chunks_x.append(feats.astype(np.float32))
            chunks_y.extend([c] * len(feats))
        return np.vstack(chunks_x), np.array(chunks_y, dtype=np.int32)

    print(f"Extracting CIFAR-100 features (ResNet50, {n_classes} classes)...")
    X_tr_img, y_tr_img = collect(train_ds, S_tr, n_classes)
    X_te_img, y_te_img = collect(test_ds,  S_te, n_classes)
    print(f"  CIFAR train: {X_tr_img.shape}  test: {X_te_img.shape}")
    return X_tr_img, y_tr_img, X_te_img, y_te_img, class_names


def fit_stitch_map(
    X_text_tr: np.ndarray,
    X_img_tr: np.ndarray,
    y_tr: np.ndarray,
    alpha: float,
) -> dict:
    # fit a ridge regression map from normalised text reps to normalised image reps
    text_chunks, img_chunks = [], []
    for c in np.unique(y_tr):
        t_idx = np.where(y_tr == c)[0]
        i_idx = np.where(y_tr == c)[0]
        n = min(len(t_idx), len(i_idx))
        text_chunks.append(X_text_tr[t_idx[:n]])
        img_chunks.append(X_img_tr[i_idx[:n]])

    X_src = np.vstack(text_chunks)
    X_tgt = np.vstack(img_chunks)

    src_mean = X_src.mean(axis=0, keepdims=True)
    src_std  = X_src.std(axis=0, keepdims=True) + 1e-8
    X_src_n = (X_src - src_mean) / src_std

    tgt_mean = X_tgt.mean(axis=0, keepdims=True)
    tgt_std  = X_tgt.std(axis=0, keepdims=True) + 1e-8
    X_tgt_n = (X_tgt - tgt_mean) / tgt_std

    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X_src_n, X_tgt_n)

    return {
        "coef": reg.coef_.astype(np.float32),
        "intercept": reg.intercept_.astype(np.float32),
        "src_mean": src_mean.astype(np.float32).squeeze(),
        "src_std":  src_std.astype(np.float32).squeeze(),
        "tgt_mean": tgt_mean.astype(np.float32).squeeze(),
        "tgt_std":  tgt_std.astype(np.float32).squeeze(),
    }


def apply_stitch_map(W: dict, X_text: np.ndarray) -> np.ndarray:
    X_n = (X_text - W["src_mean"]) / W["src_std"]
    Z_n = X_n @ W["coef"].T + W["intercept"]
    return Z_n * W["tgt_std"] + W["tgt_mean"]


def stitch_accuracy(
    X_text_te: np.ndarray,
    y_te: np.ndarray,
    X_img_tr: np.ndarray,
    y_img_tr: np.ndarray,
    W: dict,
) -> tuple[float, np.ndarray]:
    # map text representations into image space and evaluate 1-NN accuracy
    X_mapped = apply_stitch_map(W, X_text_te)

    nn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", algorithm="brute")
    nn.fit(X_img_tr, y_img_tr)
    y_pred = nn.predict(X_mapped)

    overall = float((y_pred == y_te).mean())

    classes = np.unique(y_te)
    per_class = np.array([
        (y_pred[y_te == c] == c).mean() for c in classes
    ])
    return overall, per_class


def multiclass_top_k_dims(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    # select top-k dimensions by mean absolute OvR logistic regression coefficient
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", multi_class="ovr")
    clf.fit(X, y)
    return np.argsort(np.abs(clf.coef_).mean(axis=0))[::-1][:k]


if __name__ == "__main__":
    DEVICE = "cuda:0"
    LAYER = None
    MAX_CONCEPTS = MAX_CONCEPTS
    SAMPLES_PER_CONCEPT = SAMPLES_PER_CONCEPT
    LOGREG_TOP_K = LOGREG_TOP_K
    TAU = TAU
    RIDGE_ALPHA = RIDGE_ALPHA
    CIFAR_ROOT = CIFAR_ROOT

    layer = LAYER if LAYER is not None else get_default_sae_layer()
    K = MAX_CONCEPTS
    S = SAMPLES_PER_CONCEPT
    S_tr = S // 2
    S_te = S - S_tr

    sae_ids = layer_to_sae_ids_gemma3(layer)
    print(f"Layer {layer} | {len(sae_ids)} SAE types | device {DEVICE}")
    print(f"K={K} concept pairs | S={S} ({S_tr} train + {S_te} test) | "
          f"ridge_alpha={RIDGE_ALPHA}\n")

    out_dir = Path(f"results/model_stitching_{MODEL_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 1: extracting CIFAR-100 image features (ResNet50)...")
    X_img_tr, y_img_tr, X_img_te, y_img_te, cifar_names = extract_cifar100_features(
        n_classes=K, samples_per_class=S, root=CIFAR_ROOT,
    )

    print("\nPhase 2: collecting text concept activations...")
    datasets = get_datasets()[:K]

    pos_tr_chunks: list[np.ndarray] = []
    pos_te_chunks: list[np.ndarray] = []
    y_text_tr_list: list[int] = []
    y_text_te_list: list[int] = []
    text_concept_names: list[str] = []
    concept_idx = 0

    for dataset in datasets:
        try:
            X_tr, y_tr, _, _ = get_xy_traintest(NUM_TRAIN_TEXT, dataset, layer, seed=42)
        except Exception as e:
            print(f"  [skip] {dataset}: {e}")
            continue

        if len(np.unique(y_tr)) < 2:
            continue

        X_tr_np = X_tr.numpy() if hasattr(X_tr, "numpy") else np.asarray(X_tr, dtype=np.float32)
        del X_tr

        pos_idx = np.where(y_tr == 1)[0]
        if len(pos_idx) < S:
            print(f"  [skip] {dataset}: only {len(pos_idx)} positive examples (need {S})")
            continue

        rng = np.random.default_rng(42 + concept_idx)
        chosen = rng.choice(pos_idx, S, replace=False)
        X_pos = X_tr_np[chosen].astype(np.float32)
        del X_tr_np

        pos_tr_chunks.append(X_pos[:S_tr])
        pos_te_chunks.append(X_pos[S_tr:])
        y_text_tr_list.extend([concept_idx] * S_tr)
        y_text_te_list.extend([concept_idx] * S_te)
        text_concept_names.append(dataset)
        concept_idx += 1

        if concept_idx >= K:
            break

    if concept_idx < 10:
        print(f"Only {concept_idx} concepts collected, need ≥10. Exiting.")
    else:
        K_actual = concept_idx
        X_img_tr = X_img_tr[y_img_tr < K_actual]
        y_img_tr = y_img_tr[y_img_tr < K_actual]
        X_img_te = X_img_te[y_img_te < K_actual]
        y_img_te = y_img_te[y_img_te < K_actual]

        X_text_tr_raw = np.vstack(pos_tr_chunks)
        X_text_te_raw = np.vstack(pos_te_chunks)
        y_text_tr = np.array(y_text_tr_list, dtype=np.int32)
        y_text_te = np.array(y_text_te_list, dtype=np.int32)
        del pos_tr_chunks, pos_te_chunks

        print(f"\nText: {X_text_tr_raw.shape[0]} train, {X_text_te_raw.shape[0]} test examples")
        print(f"CIFAR: {X_img_tr.shape[0]} train, {X_img_te.shape[0]} test examples")
        print(f"Concepts matched: {K_actual}\n")

        print("Phase 3: fitting multiclass logreg for top-k dim selection...")
        top_k_idx = multiclass_top_k_dims(X_text_tr_raw, y_text_tr, LOGREG_TOP_K)
        X_text_tr_lr = X_text_tr_raw[:, top_k_idx]
        X_text_te_lr = X_text_te_raw[:, top_k_idx]
        print(f"  Selected {LOGREG_TOP_K} dims.\n")

        all_results = []

        for rep_name, X_tr_rep, X_te_rep in [
            ("raw",    X_text_tr_raw, X_text_te_raw),
            ("logreg", X_text_tr_lr,  X_text_te_lr),
        ]:
            print(f"Stitching [{rep_name}] {X_tr_rep.shape[1]}-d → 2048-d img...")
            W = fit_stitch_map(X_tr_rep, X_img_tr, y_text_tr, RIDGE_ALPHA)
            acc, per_cls = stitch_accuracy(X_te_rep, y_text_te, X_img_tr, y_img_tr, W)
            print(f"  Stitching accuracy: {acc:.4f}  (chance={1/K_actual:.4f})")
            all_results.append({
                "rep": rep_name, "n_dims": int(X_tr_rep.shape[1]),
                "stitch_accuracy": acc,
                "stitch_accuracy_vs_chance": acc - 1 / K_actual,
                "per_class_acc_mean": float(per_cls.mean()),
                "per_class_acc_std": float(per_cls.std()),
            })

        del X_text_tr_lr, X_text_te_lr

        print()
        for sae_id in sae_ids:
            sae_type = sae_id.split("/")[-1]
            print(f"Loading SAE [{sae_type}]...")
            try:
                sae = sae_id_to_sae_gemma3(sae_id, DEVICE)
            except Exception as e:
                print(f"  [skip] {sae_id}: {e}")
                continue

            try:
                X_tr_t = torch.from_numpy(X_text_tr_raw).float()
                X_te_t = torch.from_numpy(X_text_te_raw).float()
                Z_tr = encode_through_sae(sae, X_tr_t, DEVICE)
                Z_te = encode_through_sae(sae, X_te_t, DEVICE)
                del X_tr_t, X_te_t
            except Exception as e:
                print(f"  [skip] {sae_id}: encode failed — {e}")
                del sae; gc.collect()
                if DEVICE.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue

            del sae; gc.collect()
            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()

            fire_frac = (Z_tr > 0).mean(axis=0)
            mask = fire_frac > TAU
            if mask.sum() < 2:
                mask = fire_frac > 0.01
            Z_tr_f = Z_tr[:, mask].astype(np.float32)
            Z_te_f = Z_te[:, mask].astype(np.float32)
            del Z_tr, Z_te; gc.collect()

            n_latents = int(mask.sum())
            print(f"  [{sae_type}] {n_latents} latents after τ filter → stitching...")
            W = fit_stitch_map(Z_tr_f, X_img_tr, y_text_tr, RIDGE_ALPHA)
            acc, per_cls = stitch_accuracy(Z_te_f, y_text_te, X_img_tr, y_img_tr, W)
            print(f"  Stitching accuracy: {acc:.4f}  (chance={1/K_actual:.4f})")

            all_results.append({
                "rep": sae_type, "n_dims": n_latents,
                "stitch_accuracy": acc,
                "stitch_accuracy_vs_chance": acc - 1 / K_actual,
                "per_class_acc_mean": float(per_cls.mean()),
                "per_class_acc_std": float(per_cls.std()),
            })

            del Z_tr_f, Z_te_f; gc.collect()

        df = pd.DataFrame(all_results).sort_values("stitch_accuracy", ascending=False)
        csv_path = out_dir / "results.csv"
        df.to_csv(csv_path, index=False)

        summary = {
            "layer": layer,
            "K_concepts": K_actual,
            "samples_per_concept": S,
            "logreg_top_k": LOGREG_TOP_K,
            "tau": TAU,
            "ridge_alpha": RIDGE_ALPHA,
            "chance_accuracy": 1 / K_actual,
            "text_concept_names": text_concept_names,
            "cifar_class_names": cifar_names,
            "results": all_results,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved → {csv_path}")

        print(f"\n{'Rep':<30} {'Dims':>7} {'Stitch Acc':>12} {'vs Chance':>12}")
        print("-" * 65)
        for row in sorted(all_results, key=lambda r: r["stitch_accuracy"], reverse=True):
            print(f"{row['rep']:<30} {row['n_dims']:>7} "
                  f"{row['stitch_accuracy']:>12.4f} {row['stitch_accuracy_vs_chance']:>+12.4f}")
        print(f"\nChance baseline: {1/K_actual:.4f} ({K_actual} classes)")
