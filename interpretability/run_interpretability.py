import json
import os
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from utils.autointerp import (
    build_autointerp_prompt,
    get_max_activating_examples,
    get_top_k_latent_indices,
    load_and_encode_normal,
    load_and_encode_ood,
    single_latent_auc,
    top_k_probe_auc,
)
from utils.data import (
    MODEL_NAME,
    get_default_sae_layer,
    get_numbered_binary_tags,
)
from utils.sae import layer_to_sae_ids_gemma3, sae_id_to_sae_gemma3

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

INTERP_DATA_DIR = f"data/interpretability_{MODEL_NAME}"


def _mkdirs():
    for sub in ("4.1_pruning", "4.2_latent_investigation", "4.3_dataset_quality"):
        os.makedirs(os.path.join(INTERP_DATA_DIR, sub), exist_ok=True)


def _metadata(dataset: str, layer: int, sae_id: str, section: str) -> dict:
    return {
        "dataset":      dataset,
        "layer":        layer,
        "sae_id":       sae_id,
        "section":      section,
        "model":        MODEL_NAME,
        "generated_at": datetime.now().isoformat(),
    }


OOD_TASK_METADATA = {
    "7_hist_fig_ispolitician": (
        "Classify whether a named historical figure is a politician.",
        "Names are replaced with syntactical alterations or cartoon characters.",
    ),
    "66_living-room": (
        "Classify whether the phrase 'living room' appears in an English sentence.",
        "Sentences are translated to French.",
    ),
    "90_glue_qnli": (
        "Classify whether a sentence entails the answer to a given question (QNLI).",
        "Sentences are from the GLUE-X OOD split (harder distribution).",
    ),
}


def prepare_section_4_1(layer: int, sae_id: str, sae, device: str, k: int = 8):
    # build per-latent autointerp prompts and OOD AUC curves for the pruning experiment
    print("\n" + "=" * 60)
    print("Section 4.1 — Preparing OOD Pruning data")
    print("=" * 60)

    for dataset, (task_desc, ood_desc) in OOD_TASK_METADATA.items():
        out_path = os.path.join(INTERP_DATA_DIR, "4.1_pruning", f"{dataset}.json")
        if os.path.exists(out_path):
            print(f"  Skipping {dataset} (already prepared)")
            continue

        print(f"\n  Dataset: {dataset}")

        try:
            (X_train_sae, y_train,
             X_test_sae,  y_test_ood,
             prompts_train, _) = load_and_encode_ood(
                dataset, layer, sae, device
            )
        except FileNotFoundError as e:
            print(f"    Skipping — {e}")
            continue

        top_idxs = get_top_k_latent_indices(X_train_sae, y_train, k)
        print(f"    Top {k} latent indices: {top_idxs.tolist()}")

        latent_data = []
        for rank, lidx in enumerate(top_idxs):
            k1_auc_id  = single_latent_auc(X_train_sae, y_train,
                                            X_train_sae, y_train, int(lidx))
            k1_auc_ood = single_latent_auc(X_train_sae, y_train,
                                            X_test_sae,  y_test_ood, int(lidx))
            examples = get_max_activating_examples(
                X_train_sae, int(lidx), prompts_train, n=10
            )
            prompt_payload = build_autointerp_prompt(examples, int(lidx))

            latent_data.append({
                "latent_idx":       int(lidx),
                "rank_by_diff":     rank,
                "k1_auc_id":        float(k1_auc_id),
                "k1_auc_ood":       float(k1_auc_ood),
                "examples":         [{"text": t, "activation": a} for t, a in examples],
                "autointerp_prompt": prompt_payload,
                "description":      None,
            })

        ood_auc_by_k = {}
        for n_keep in range(1, k + 1):
            subset = top_idxs[:n_keep]
            auc = top_k_probe_auc(X_train_sae, y_train, X_test_sae, y_test_ood, subset)
            ood_auc_by_k[n_keep] = float(auc)
            print(f"      k={n_keep} (diff order): OOD AUC = {auc:.4f}")

        ranking_prompt_stub = {
            "note": "Rebuild with actual descriptions after autointerp step.",
            "task_description": task_desc,
            "ood_transform":    ood_desc,
            "latent_order":     [int(i) for i in top_idxs],
        }

        payload = {
            "metadata":          _metadata(dataset, layer, sae_id, "4.1"),
            "task_description":  task_desc,
            "ood_transform":     ood_desc,
            "latents":           latent_data,
            "ood_auc_by_k_diff_order": ood_auc_by_k,
            "ranking_prompt_stub":     ranking_prompt_stub,
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"    Saved → {out_path}")


def prepare_section_4_2(
    layer: int,
    sae_id: str,
    sae,
    device: str,
    max_datasets: int | None = None,
    top_latents: int = 128,
):
    # find the single best latent per dataset and build its autointerp prompt
    print("\n" + "=" * 60)
    print("Section 4.2 — Preparing Latent Investigation data")
    print("=" * 60)

    datasets = get_numbered_binary_tags()
    if max_datasets:
        datasets = datasets[:max_datasets]

    for dataset in tqdm(datasets, desc="Datasets"):
        out_path = os.path.join(INTERP_DATA_DIR, "4.2_latent_investigation",
                                f"{dataset}.json")
        if os.path.exists(out_path):
            continue

        try:
            (X_train_sae, y_train,
             X_test_sae,  y_test,
             prompts_train, _) = load_and_encode_normal(dataset, layer, sae, device)
        except (FileNotFoundError, ValueError, KeyError) as e:
            tqdm.write(f"  Skipping {dataset}: {e}")
            continue

        candidate_idxs = get_top_k_latent_indices(X_train_sae, y_train, top_latents)

        latent_aucs = []
        for lidx in candidate_idxs:
            auc = single_latent_auc(X_train_sae, y_train,
                                     X_test_sae,  y_test, int(lidx))
            latent_aucs.append((int(lidx), float(auc)))

        latent_aucs.sort(key=lambda x: x[1], reverse=True)
        best_latent_idx, best_auc = latent_aucs[0]

        examples = get_max_activating_examples(
            X_train_sae, best_latent_idx, prompts_train, n=10
        )
        prompt_payload = build_autointerp_prompt(examples, best_latent_idx)

        payload = {
            "metadata": _metadata(dataset, layer, sae_id, "4.2"),
            "best_latent": {
                "latent_idx":        best_latent_idx,
                "k1_auc":            best_auc,
                "examples":          [{"text": t, "activation": a} for t, a in examples],
                "autointerp_prompt": prompt_payload,
                "description":       None,
            },
            "all_candidate_latents": [
                {"latent_idx": idx, "k1_auc": auc}
                for idx, auc in latent_aucs
            ],
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        tqdm.write(f"  {dataset}: best latent={best_latent_idx}  AUC={best_auc:.3f}")


QUALITY_DATASETS = ["87_glue_cola", "110_aimade_humangpt3"]


def _find_disagreements(
    X_sae: np.ndarray,
    y: np.ndarray,
    prompts: list[str],
    latent_idx: int,
    threshold_percentile: float = 90,
    n_show: int = 10,
) -> list[dict]:
    # examples where a latent fires strongly but disagrees with the label
    acts = X_sae[:, latent_idx]
    threshold = np.percentile(acts, threshold_percentile)
    fires = acts >= threshold

    mean_pos = acts[y == 1].mean() if (y == 1).any() else 0.0
    mean_neg = acts[y == 0].mean() if (y == 0).any() else 0.0
    if mean_neg > mean_pos:
        disagree_mask = fires & (y == 1)
    else:
        disagree_mask = fires & (y == 0)

    disagree_idxs = np.where(disagree_mask)[0]
    disagree_idxs = sorted(disagree_idxs, key=lambda i: acts[i], reverse=True)[:n_show]

    return [
        {"text": prompts[i], "activation": float(acts[i]), "label": int(y[i])}
        for i in disagree_idxs
    ]


def prepare_section_4_3(layer: int, sae_id: str, sae, device: str):
    # identify label-disagreement examples using the top latent for dataset quality analysis
    print("\n" + "=" * 60)
    print("Section 4.3 — Preparing Dataset Quality data")
    print("=" * 60)

    for dataset in QUALITY_DATASETS:
        out_path = os.path.join(INTERP_DATA_DIR, "4.3_dataset_quality",
                                f"{dataset}.json")
        if os.path.exists(out_path):
            print(f"  Skipping {dataset} (already prepared)")
            continue

        print(f"\n  Dataset: {dataset}")

        try:
            (X_train_sae, y_train,
             X_test_sae,  y_test,
             prompts_train, prompts_test) = load_and_encode_normal(
                dataset, layer, sae, device
            )
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"    Skipping — {e}")
            continue

        X_all       = np.concatenate([X_train_sae, X_test_sae], axis=0)
        y_all       = np.concatenate([y_train,     y_test],     axis=0)
        prompts_all = prompts_train + prompts_test

        top_idx = int(get_top_k_latent_indices(X_train_sae, y_train, k=1)[0])
        k1_auc  = single_latent_auc(X_train_sae, y_train, X_test_sae, y_test, top_idx)
        print(f"    Top latent: {top_idx}   k=1 AUC: {k1_auc:.4f}")

        examples = get_max_activating_examples(
            X_train_sae, top_idx, prompts_train, n=10
        )
        prompt_payload = build_autointerp_prompt(examples, top_idx)

        disagreements = _find_disagreements(
            X_all, y_all, prompts_all, top_idx,
            threshold_percentile=90, n_show=10,
        )

        payload = {
            "metadata": _metadata(dataset, layer, sae_id, "4.3"),
            "top_latent": {
                "latent_idx":        top_idx,
                "k1_auc":            float(k1_auc),
                "examples":          [{"text": t, "activation": a} for t, a in examples],
                "autointerp_prompt": prompt_payload,
                "description":       None,
            },
            "label_disagreements": disagreements,
        }

        if dataset == "110_aimade_humangpt3":
            last_chars_ai    = [p[-1] for p, lbl in zip(prompts_all, y_all)
                                if lbl == 1 and p]
            last_chars_human = [p[-1] for p, lbl in zip(prompts_all, y_all)
                                if lbl == 0 and p]
            payload["last_token_distribution"] = {
                "ai_top10":    Counter(last_chars_ai).most_common(10),
                "human_top10": Counter(last_chars_human).most_common(10),
            }
            print(f"    AI last-chars:    {Counter(last_chars_ai).most_common(5)}")
            print(f"    Human last-chars: {Counter(last_chars_human).most_common(5)}")

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"    Saved → {out_path}")


def _resolve_sae_id(sae_id_arg: str | None, layer: int) -> str:
    # pick a sensible default SAE id if none is specified (prefers topk)
    if sae_id_arg:
        return sae_id_arg
    ids = layer_to_sae_ids_gemma3(layer)
    if not ids:
        raise RuntimeError(
            f"No trained SAEs found for layer {layer}. "
            "Run setup/train_sae.py first."
        )
    for candidate in ids:
        if "topk" in candidate:
            return candidate
    return ids[0]


if __name__ == "__main__":
    DEVICE = "cuda:0"
    SECTION = "all"       # one of: 4.1, 4.2, 4.3, all
    SAE_ID = None         # set to e.g. "layer_22/width_16k/topk_32", or None to auto-resolve
    MAX_DATASETS = None   # (section 4.2) limit to first N datasets; None = all
    TOP_LATENTS = 128     # (section 4.2) candidate latents per dataset
    K = 8                 # (section 4.1) top latents for OOD pruning sweep

    layer  = int(get_default_sae_layer(MODEL_NAME))
    sae_id = _resolve_sae_id(SAE_ID, layer)

    print(f"Model  : {MODEL_NAME}")
    print(f"Layer  : {layer}")
    print(f"SAE ID : {sae_id}")
    print(f"Device : {DEVICE}")

    print(f"\nLoading SAE {sae_id} ...")
    sae = sae_id_to_sae_gemma3(sae_id, DEVICE)
    sae.eval()

    _mkdirs()

    if SECTION in ("4.1", "all"):
        prepare_section_4_1(layer, sae_id, sae, DEVICE, k=K)

    if SECTION in ("4.2", "all"):
        prepare_section_4_2(layer, sae_id, sae, DEVICE,
                             max_datasets=MAX_DATASETS,
                             top_latents=TOP_LATENTS)

    if SECTION in ("4.3", "all"):
        prepare_section_4_3(layer, sae_id, sae, DEVICE)

    print(f"\nCompute step done. Prompt payloads saved to: {INTERP_DATA_DIR}/")
    print("Next: run interpretability/run_llm_interp.py")
