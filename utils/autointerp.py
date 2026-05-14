import json
import re
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.simplefilter("ignore", category=ConvergenceWarning)


def encode_through_sae(sae, X: torch.Tensor, device: str, batch_size: int = 256) -> np.ndarray:
    # SAE activation files are deleted after probe training to save disk space,
    # so we re-encode on the fly whenever we need them for interpretability work
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)

    parts = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size].to(device)
        with torch.no_grad():
            encoded = sae.encode(batch)
        parts.append(encoded.float().cpu())

    return torch.cat(parts, dim=0).numpy()


def load_and_encode_normal(dataset: str, layer: int, sae, device: str, max_amt: int = 5000):
    # load in-distribution activations, encode through SAE, and return train/test split with prompts
    from utils.data import (
        MODEL_NAME,
        get_dataset_sizes,
        get_train_test_indices,
        get_xvals,
        get_yvals,
        read_numbered_dataset_df,
    )

    dataset_sizes = get_dataset_sizes()
    size = dataset_sizes[dataset]
    num_train = min(size - 100, 1024)

    X = get_xvals(dataset, layer, MODEL_NAME).float()
    y = get_yvals(dataset)
    prompts = read_numbered_dataset_df(dataset)["prompt"].tolist()

    N = min(len(X), max_amt)
    X, y, prompts = X[:N], y[:N], prompts[:N]

    num_test = N - num_train - 1
    train_idx, test_idx = get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=42)

    X_train_sae = encode_through_sae(sae, X[train_idx], device)
    X_test_sae  = encode_through_sae(sae, X[test_idx],  device)

    return (
        X_train_sae,
        y[train_idx].astype(int),
        X_test_sae,
        y[test_idx].astype(int),
        [prompts[i] for i in train_idx],
        [prompts[i] for i in test_idx],
    )


def load_and_encode_ood(dataset: str, layer: int, sae, device: str, max_train: int = 1024):
    # load ID train activations and OOD test activations, encode both through SAE
    from utils.data import (
        MODEL_NAME,
        get_dataset_sizes,
        get_train_test_indices,
        get_xvals,
        get_yvals,
        read_numbered_dataset_df,
    )
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    dataset_sizes = get_dataset_sizes()
    size = dataset_sizes.get(dataset, 0)
    num_train = min(max(size - 100, 0), max_train)

    X_id   = get_xvals(dataset, layer, MODEL_NAME).float()
    y_id   = get_yvals(dataset)
    prompts_id = read_numbered_dataset_df(dataset)["prompt"].tolist()

    N = min(len(X_id), 5000)
    X_id, y_id, prompts_id = X_id[:N], y_id[:N], prompts_id[:N]

    num_test_dummy = max(N - num_train - 1, 0)
    train_idx, _ = get_train_test_indices(y_id, num_train, num_test_dummy, pos_ratio=0.5, seed=42)

    X_train_sae   = encode_through_sae(sae, X_id[train_idx], device)
    y_train       = y_id[train_idx].astype(int)
    prompts_train = [prompts_id[i] for i in train_idx]

    # OOD test examples come from a separate directory with distribution-shifted data
    ood_act_path = (
        f"data/model_activations_{MODEL_NAME}_OOD/"
        f"{dataset}_blocks.{layer}.hook_resid_post.pt"
    )
    X_ood = torch.load(ood_act_path, weights_only=False).float()

    ood_csv = f"data/OOD data/{dataset}_OOD.csv"
    df_ood  = pd.read_csv(ood_csv)
    le = LabelEncoder()
    y_ood = le.fit_transform(df_ood["target"].values).astype(int)
    prompts_ood = df_ood["prompt"].tolist()

    X_test_sae = encode_through_sae(sae, X_ood, device)

    return (
        X_train_sae, y_train,
        X_test_sae,  y_ood,
        prompts_train, prompts_ood,
    )


def get_top_k_latent_indices(X_train_sae: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    # rank latents by |mean_pos - mean_neg| — this is how the paper selects features
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    mean_pos = X_train_sae[pos_mask].mean(axis=0)
    mean_neg = X_train_sae[neg_mask].mean(axis=0)
    diff = np.abs(mean_pos - mean_neg)
    return np.argsort(diff)[::-1][:k]


def get_max_activating_examples(X_sae: np.ndarray, latent_idx: int, prompts: list[str], n: int = 10) -> list[tuple[str, float]]:
    # return the n prompts that most strongly activate this latent
    acts = X_sae[:, latent_idx]
    top_n = np.argsort(acts)[::-1][:n]
    return [(prompts[i], float(acts[i])) for i in top_n]


# system prompts for the Claude API calls

_AUTOINTERP_SYSTEM = (
    "You are a mechanistic interpretability researcher analysing sparse autoencoder "
    "latents. Given the text examples that most strongly activate a latent, produce a "
    "concise, specific 1-2 sentence description of the linguistic or conceptual pattern "
    "the latent represents. Do not hedge; be direct."
)

_RANKING_SYSTEM = (
    "You are a mechanistic interpretability researcher evaluating whether sparse "
    "autoencoder features are useful for a classification task under distribution shift. "
    "Rank the features by how genuinely relevant they are to the task semantics, "
    "downranking any that appear to rely on spurious correlations that would break under "
    "the described OOD transformation."
)


def build_autointerp_prompt(examples: list[tuple[str, float]], latent_idx: int, max_example_chars: int = 300) -> dict:
    # format top activating examples into a prompt for the autointerp Claude call
    formatted = "\n".join(
        f"  [{act:.3f}] {text[:max_example_chars].strip()}"
        for text, act in examples
    )
    user_msg = (
        f"Latent index: {latent_idx}\n\n"
        "Top activating examples (format: [activation] text):\n"
        f"{formatted}\n\n"
        "Description:"
    )
    return {"system": _AUTOINTERP_SYSTEM, "user": user_msg}


def build_ranking_prompt(latent_descriptions: dict[int, str], task_description: str, ood_transform: str) -> dict:
    # build the LLM prompt asking it to rank latents by OOD robustness
    items = list(latent_descriptions.items())
    latent_list = "\n".join(
        f"  {i+1}. (latent {idx}) {desc}"
        for i, (idx, desc) in enumerate(items)
    )
    user_msg = (
        f"Task: {task_description}\n"
        f"OOD transformation: {ood_transform}\n\n"
        f"Candidate latents:\n{latent_list}\n\n"
        "Rank the latents from most relevant to least relevant for this task under the "
        "OOD transformation. Return ONLY a JSON object of the form "
        '{"ranking": [<1-based positions in order of relevance>]}. '
        'Example: {"ranking": [3, 1, 5, 2, 4, 6, 7, 8]}'
    )
    return {"system": _RANKING_SYSTEM, "user": user_msg}


def parse_ranking_response(response_text: str, items: list[tuple[int, str]]) -> list[int]:
    # parse the LLM ranking JSON; fall back to original order if parsing fails
    match = re.search(r'\{[^{}]*"ranking"[^{}]*\}', response_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            positions = data["ranking"]
            return [items[pos - 1][0] for pos in positions if 1 <= pos <= len(items)]
        except (KeyError, IndexError, json.JSONDecodeError):
            pass
    return [idx for idx, _ in items]


def single_latent_auc(
    X_train_sae: np.ndarray,
    y_train: np.ndarray,
    X_test_sae: np.ndarray,
    y_test: np.ndarray,
    latent_idx: int,
) -> float:
    # fit a single-feature logistic regression on one latent and evaluate AUC
    Xtr = X_train_sae[:, latent_idx].reshape(-1, 1)
    Xte = X_test_sae[:, latent_idx].reshape(-1, 1)

    n = len(Xtr)
    val_start = int(0.8 * n)
    if val_start < 2 or n - val_start < 2:
        clf = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=1000, random_state=42)
        clf.fit(Xtr, y_train)
    else:
        best_C, best_val = 1.0, -1.0
        for C in np.logspace(5, -5, 10):
            clf = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=1000, random_state=42)
            clf.fit(Xtr[:val_start], y_train[:val_start])
            try:
                score = roc_auc_score(y_train[val_start:], clf.predict_proba(Xtr[val_start:])[:, 1])
            except ValueError:
                continue
            if score > best_val:
                best_val, best_C = score, C
        clf = LogisticRegression(penalty="l1", solver="saga", C=best_C, max_iter=1000, random_state=42)
        clf.fit(Xtr, y_train)

    try:
        return roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1])
    except ValueError:
        return float("nan")


def top_k_probe_auc(
    X_train_sae: np.ndarray,
    y_train: np.ndarray,
    X_test_sae: np.ndarray,
    y_test: np.ndarray,
    latent_indices: np.ndarray,
) -> float:
    # fit a logistic regression on top-k latents and evaluate AUC on the test set
    Xtr = X_train_sae[:, latent_indices]
    Xte = X_test_sae[:, latent_indices]

    n = len(Xtr)
    val_start = int(0.8 * n)
    if val_start < 2 or n - val_start < 2:
        clf = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=1000, random_state=42)
        clf.fit(Xtr, y_train)
    else:
        best_C, best_val = 1.0, -1.0
        for C in np.logspace(5, -5, 10):
            clf = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=1000, random_state=42)
            clf.fit(Xtr[:val_start], y_train[:val_start])
            try:
                score = roc_auc_score(y_train[val_start:], clf.predict_proba(Xtr[val_start:])[:, 1])
            except ValueError:
                continue
            if score > best_val:
                best_val, best_C = score, C
        clf = LogisticRegression(penalty="l1", solver="saga", C=best_C, max_iter=1000, random_state=42)
        clf.fit(Xtr, y_train)

    try:
        return roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1])
    except ValueError:
        return float("nan")
