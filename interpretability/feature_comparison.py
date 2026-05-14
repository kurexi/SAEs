import json
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from utils.autointerp import (
    encode_through_sae,
    get_top_k_latent_indices,
)
from utils.data import (
    MODEL_NAME,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_train_test_indices,
    get_xvals,
    get_yvals,
)
from utils.sae import layer_to_sae_ids_gemma3, sae_id_to_sae_gemma3

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

RESULTS_DIR = f"results/interpretability_comparison_{MODEL_NAME}"

SEMANTIC_GROUPS = {
    "historical_figures": [
        "5_hist_fig_ismale", "6_hist_fig_isamerican", "7_hist_fig_ispolitician",
    ],
    "news_headlines": [
        "21_headline_istrump", "22_headline_isobama", "23_headline_ischina",
        "24_headline_isiran", "26_headline_isfrontpage",
    ],
    "wikidata_person": [
        "56_wikidatasex_or_gender", "57_wikidatais_alive", "58_wikidatapolitical_party",
        "59_wikidata_occupation_isjournalist", "60_wikidata_occupation_isathlete",
        "61_wikidata_occupation_isactor", "62_wikidata_occupation_ispolitician",
        "63_wikidata_occupation_issinger", "64_wikidata_occupation_isresearcher",
    ],
    "nyc_boroughs": [
        "114_nyc_borough_Manhattan", "115_nyc_borough_Brooklyn", "116_nyc_borough_Bronx",
    ],
    "us_states": [
        "117_us_state_FL", "118_us_state_CA", "119_us_state_TX",
    ],
    "us_timezones": [
        "120_us_timezone_Chicago", "121_us_timezone_New_York", "122_us_timezone_Los_Angeles",
    ],
    "world_countries": [
        "123_world_country_United_Kingdom", "124_world_country_United_States",
        "125_world_country_Italy",
    ],
    "art_types": [
        "126_art_type_book", "127_art_type_song", "128_art_type_movie",
    ],
    "glue_nli": [
        "136_glue_mnli_entailment", "137_glue_mnli_neutral", "138_glue_mnli_contradiction",
    ],
    "glue_misc": [
        "87_glue_cola", "90_glue_qnli", "91_glue_qqp", "92_glue_sst2",
    ],
    "cancer_categories": [
        "142_cancer_cat_Thyroid_Cancer", "143_cancer_cat_Lung_Cancer",
        "144_cancer_cat_Colon_Cancer",
    ],
    "disease_classes": [
        "145_disease_class_digestive system diseases",
        "146_disease_class_cardiovascular diseases",
        "147_disease_class_nervous system diseases",
    ],
    "twitter_emotions": [
        "148_twt_emotion_worry", "149_twt_emotion_happiness", "150_twt_emotion_sadness",
    ],
    "it_tickets": [
        "151_it_tick_HR Support", "152_it_tick_Hardware",
        "153_it_tick_Administrative rights",
    ],
    "athlete_sports": [
        "154_athlete_sport_football", "155_athlete_sport_basketball",
        "156_athlete_sport_baseball",
    ],
    "code_languages": [
        "158_code_C", "159_code_Python", "160_code_HTML",
    ],
    "news_topics": [
        "139_news_class_Politics", "140_news_class_Technology",
        "141_news_class_Entertainment",
    ],
    "agnews": [
        "161_agnews_0", "162_agnews_1", "163_agnews_2",
    ],
    "ethics_categories": [
        "48_cm_correct", "49_cm_isshort", "50_deon_isvalid",
        "51_just_is", "52_virtue_is",
    ],
    "temporal_reasoning": [
        "42_temp_sense", "130_temp_cat_Frequency",
        "131_temp_cat_Typical Time", "132_temp_cat_Event Ordering",
    ],
    "context_types": [
        "133_context_type_Causality", "134_context_type_Belief_states",
        "135_context_type_Event_duration",
    ],
    "phrases_in_context": [
        "65_high-school", "66_living-room", "67_social-security", "68_credit-card",
        "69_blood-pressure", "70_prime-factors", "71_social-media", "72_gene-expression",
        "73_control-group", "74_magnetic-field", "75_cell-lines", "76_trial-court",
        "77_second-derivative", "78_north-america", "79_human-rights", "80_side-effects",
        "81_public-health", "82_federal-government", "83_third-party", "84_clinical-trials",
        "85_mental-health",
    ],
}

DATASET_TO_GROUP = {
    ds: grp
    for grp, datasets in SEMANTIC_GROUPS.items()
    for ds in datasets
}


def gini_coefficient(x: np.ndarray) -> float:
    # 0 = uniform, 1 = fully concentrated in one feature
    x = np.abs(x).astype(float)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2.0 * np.dot(idx, x)) / (n * x.sum()) - (n + 1.0) / n)


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def train_logreg(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    # cross-validate regularisation strength on the last 20% of training data
    n = len(X_train)
    val_start = int(0.8 * n)

    best_C, best_val = 1.0, -1.0
    if val_start >= 2 and n - val_start >= 2:
        for C in np.logspace(5, -5, 10):
            clf = LogisticRegression(
                penalty="l2", C=C, solver="lbfgs",
                max_iter=1000, random_state=42,
            )
            clf.fit(X_train[:val_start], y_train[:val_start])
            try:
                score = roc_auc_score(
                    y_train[val_start:],
                    clf.predict_proba(X_train[val_start:])[:, 1],
                )
            except ValueError:
                continue
            if score > best_val:
                best_val, best_C = score, C

    clf = LogisticRegression(
        penalty="l2", C=best_C, solver="lbfgs",
        max_iter=1000, random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def top_k_logreg_dims(clf: LogisticRegression, k: int) -> np.ndarray:
    w = clf.coef_[0]
    return np.argsort(np.abs(w))[::-1][:k]


def compute_dataset_metrics(
    dataset: str,
    layer: int,
    sae,
    device: str,
    k: int,
    max_train: int = 1024,
) -> dict | None:
    # compute Gini selectivity and Jaccard feature reuse for SAE vs logistic-regression dims
    try:
        X_raw = get_xvals(dataset, layer, MODEL_NAME).float().numpy()
        y     = get_yvals(dataset)
    except FileNotFoundError:
        return None

    N = min(len(X_raw), 5000)
    X_raw, y = X_raw[:N], y[:N]

    num_train = min(max_train, N - 100)
    num_test  = N - num_train - 1
    if num_train < 10 or num_test < 10:
        return None

    train_idx, test_idx = get_train_test_indices(y, num_train, num_test,
                                                  pos_ratio=0.5, seed=42)
    X_train_raw = X_raw[train_idx]
    X_test_raw  = X_raw[test_idx]
    y_train     = y[train_idx].astype(int)
    y_test      = y[test_idx].astype(int)

    X_train_sae = encode_through_sae(sae, torch.tensor(X_train_raw), device)
    X_test_sae  = encode_through_sae(sae, torch.tensor(X_test_raw),  device)

    sae_top_k = get_top_k_latent_indices(X_train_sae, y_train, k).tolist()

    clf = train_logreg(X_train_raw, y_train)
    logreg_top_k = top_k_logreg_dims(clf, k).tolist()

    try:
        sae_clf = LogisticRegression(penalty="l1", solver="saga", C=1.0,
                                     max_iter=1000, random_state=42)
        sae_clf.fit(X_train_sae[:, sae_top_k], y_train)
        sae_auc = float(roc_auc_score(
            y_test, sae_clf.predict_proba(X_test_sae[:, sae_top_k])[:, 1]))
    except ValueError:
        sae_auc = float("nan")

    try:
        logreg_auc = float(roc_auc_score(
            y_test, clf.predict_proba(X_test_raw)[:, 1]))
    except ValueError:
        logreg_auc = float("nan")

    X_all_sae = np.concatenate([X_train_sae, X_test_sae], axis=0)
    X_all_raw = np.concatenate([X_train_raw, X_test_raw],  axis=0)

    sae_ginis    = [gini_coefficient(X_all_sae[:, i]) for i in sae_top_k]
    logreg_ginis = [gini_coefficient(X_all_raw[:, i]) for i in logreg_top_k]

    sae_frac_active = [
        float((X_all_sae[:, i] > 0).mean()) for i in sae_top_k
    ]
    logreg_frac_active = [
        float((X_all_raw[:, i] > X_all_raw[:, i].mean()).mean())
        for i in logreg_top_k
    ]

    return {
        "dataset":          dataset,
        "sae_ginis":        sae_ginis,
        "logreg_ginis":     logreg_ginis,
        "sae_frac_active":  sae_frac_active,
        "logreg_frac_active": logreg_frac_active,
        "sae_top_k":        sae_top_k,
        "logreg_top_k":     logreg_top_k,
        "sae_auc":          sae_auc,
        "logreg_auc":       logreg_auc,
    }


def compute_feature_reuse(dataset_results: dict[str, dict], k: int) -> list[dict]:
    # compute pairwise Jaccard overlap of top-k features within each semantic group
    rows = []
    for group, members in SEMANTIC_GROUPS.items():
        available = [m for m in members if m in dataset_results]
        if len(available) < 2:
            continue

        for ds_a, ds_b in combinations(available, 2):
            sae_j = jaccard(
                set(dataset_results[ds_a]["sae_top_k"]),
                set(dataset_results[ds_b]["sae_top_k"]),
            )
            logreg_j = jaccard(
                set(dataset_results[ds_a]["logreg_top_k"]),
                set(dataset_results[ds_b]["logreg_top_k"]),
            )
            rows.append({
                "group":     group,
                "dataset_a": ds_a,
                "dataset_b": ds_b,
                "sae_jaccard":    sae_j,
                "logreg_jaccard": logreg_j,
                "delta_jaccard":  sae_j - logreg_j,
            })
    return rows


if __name__ == "__main__":
    from utils.data import get_default_sae_layer

    DEVICE = "cuda:0"
    K = 16
    LAYER = None
    SAE_ID = None
    MAX_DATASETS = None

    layer = LAYER or int(get_default_sae_layer(MODEL_NAME))

    sae_id = SAE_ID
    if sae_id is None:
        ids = layer_to_sae_ids_gemma3(layer)
        for candidate in ids:
            if "topk" in candidate:
                sae_id = candidate
                break
        if sae_id is None:
            sae_id = ids[0] if ids else f"layer_{layer}/width_16k/topk_32"

    print(f"Model: {MODEL_NAME}  Layer: {layer}  SAE: {sae_id}  Device: {DEVICE}  k: {K}")

    sae = sae_id_to_sae_gemma3(sae_id, DEVICE)
    sae.eval()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_datasets = get_numbered_binary_tags()
    if MAX_DATASETS:
        all_datasets = all_datasets[:MAX_DATASETS]

    dataset_results: dict[str, dict] = {}
    selectivity_rows = []

    for dataset in tqdm(all_datasets, desc="Datasets"):
        result = compute_dataset_metrics(dataset, layer, sae, DEVICE, k=K)
        if result is None:
            tqdm.write(f"  Skipped: {dataset}")
            continue
        dataset_results[dataset] = result
        for rank, (sg, lg, sfa, lfa) in enumerate(zip(
            result["sae_ginis"], result["logreg_ginis"],
            result["sae_frac_active"], result["logreg_frac_active"],
        )):
            selectivity_rows.append({
                "dataset": dataset, "rank": rank,
                "group": DATASET_TO_GROUP.get(dataset, "other"),
                "sae_gini": sg, "logreg_gini": lg,
                "sae_frac_active": sfa, "logreg_frac_active": lfa,
                "sae_auc": result["sae_auc"], "logreg_auc": result["logreg_auc"],
            })
        tqdm.write(
            f"  {dataset:45s}  "
            f"SAE gini={np.mean(result['sae_ginis']):.3f}  "
            f"LR gini={np.mean(result['logreg_ginis']):.3f}  "
            f"SAE auc={result['sae_auc']:.3f}  LR auc={result['logreg_auc']:.3f}"
        )

    sel_path = os.path.join(RESULTS_DIR, "selectivity_results.csv")
    pd.DataFrame(selectivity_rows).to_csv(sel_path, index=False)
    print(f"\nSelectivity results -> {sel_path}")

    reuse_rows = compute_feature_reuse(dataset_results, k=K)
    reuse_path = os.path.join(RESULTS_DIR, "feature_reuse_results.csv")
    pd.DataFrame(reuse_rows).to_csv(reuse_path, index=False)
    print(f"Feature reuse results -> {reuse_path}")

    sel_df   = pd.DataFrame(selectivity_rows)
    reuse_df = pd.DataFrame(reuse_rows)

    mean_sae_gini    = sel_df["sae_gini"].mean()
    mean_logreg_gini = sel_df["logreg_gini"].mean()
    sae_wins_gini    = (sel_df["sae_gini"] > sel_df["logreg_gini"]).mean()

    summary = {
        "k": K, "layer": layer, "sae_id": sae_id,
        "n_datasets": sel_df["dataset"].nunique(),
        "selectivity": {
            "mean_sae_gini": float(mean_sae_gini),
            "mean_logreg_gini": float(mean_logreg_gini),
            "delta_gini": float(mean_sae_gini - mean_logreg_gini),
            "sae_wins_pct_features": float(sae_wins_gini),
        },
        "feature_reuse": {
            "n_pairs": len(reuse_df),
            "mean_sae_jaccard": float(reuse_df["sae_jaccard"].mean()) if len(reuse_df) else float("nan"),
            "mean_logreg_jaccard": float(reuse_df["logreg_jaccard"].mean()) if len(reuse_df) else float("nan"),
        },
    }

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {summary_path}")

    print(f"\nGini: SAE={mean_sae_gini:.4f}  LogReg={mean_logreg_gini:.4f}  "
          f"d={mean_sae_gini - mean_logreg_gini:+.4f}  SAE wins: {sae_wins_gini:.1%}")
