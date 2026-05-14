import os
import glob
import re

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

MODEL_NAME = "gemma-3-4b-it"
BASEPATH = "."
SAE_ACT_BASEPATH = "scratch/data"


def get_binary_df():
    # load the master probing dataset CSV and filter to binary classification tasks only
    df = pd.read_csv("data/probing_datasets_MASTER.csv")
    return df[df["Data type"] == "Binary Classification"]


def get_numbered_binary_tags():
    # return list of dataset tag strings like "5_hist_fig_ismale"
    df = get_binary_df()
    return [name.split("/")[-1].split(".")[0] for name in df["Dataset save name"]]


def read_dataset_df(dataset_tag):
    df = get_binary_df()
    dataset_save_name = df[df["Dataset Tag"] == dataset_tag]["Dataset save name"].iloc[0]
    return pd.read_csv(f"cleaned_data/{os.path.basename(dataset_save_name)}")


def read_numbered_dataset_df(numbered_dataset_tag):
    dataset_tag = "_".join(numbered_dataset_tag.split("_")[1:])
    return read_dataset_df(dataset_tag)


def get_yvals(numbered_dataset_tag):
    # encode the target column as 0/1 integers
    df = read_numbered_dataset_df(numbered_dataset_tag)
    le = LabelEncoder()
    return le.fit_transform(df["target"].values)


def get_xvals(numbered_dataset_tag, layer, model_name=MODEL_NAME):
    # load pre-saved residual stream activations for a dataset at a given layer
    if layer == "embed":
        fname = f"data/model_activations_{model_name}/{numbered_dataset_tag}_hook_embed.pt"
    else:
        fname = (
            f"data/model_activations_{model_name}/"
            f"{numbered_dataset_tag}_blocks.{layer}.hook_resid_post.pt"
        )
    return torch.load(fname, weights_only=False)


def get_xyvals(numbered_dataset_tag, layer, model_name=MODEL_NAME, MAX_AMT=1500):
    xvals = get_xvals(numbered_dataset_tag, layer, model_name)
    yvals = get_yvals(numbered_dataset_tag)
    return xvals[:MAX_AMT], yvals[:MAX_AMT]


def get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=42):
    # stratified split: sample exactly pos_ratio positives in train and test
    np.random.seed(seed)

    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    pos_train_size = int(np.ceil(pos_ratio * num_train))
    neg_train_size = num_train - pos_train_size
    pos_test_size = int(np.ceil(pos_ratio * num_test))
    neg_test_size = num_test - pos_test_size

    train_pos = np.random.choice(pos_indices, size=pos_train_size, replace=False)
    train_neg = np.random.choice(neg_indices, size=neg_train_size, replace=False)

    remaining_pos = np.setdiff1d(pos_indices, train_pos)
    remaining_neg = np.setdiff1d(neg_indices, train_neg)

    test_pos = np.random.choice(remaining_pos, size=pos_test_size, replace=False)
    test_neg = np.random.choice(remaining_neg, size=neg_test_size, replace=False)

    train_indices = np.random.permutation(np.concatenate([train_pos, train_neg]))
    test_indices = np.random.permutation(np.concatenate([test_pos, test_neg]))
    return train_indices, test_indices


def get_xy_traintest_specify(
    num_train,
    numbered_dataset_tag,
    layer,
    model_name=MODEL_NAME,
    pos_ratio=0.5,
    MAX_AMT=5000,
    seed=42,
    num_test=None,
):
    # load activations and split into stratified train/test sets with a custom pos ratio
    X, y = get_xyvals(numbered_dataset_tag, layer, model_name, MAX_AMT=MAX_AMT)
    if num_test is None:
        num_test = X.shape[0] - num_train - 1
    if num_train + min(100, num_test) > X.shape[0]:
        raise ValueError(
            f"Requested {num_train + 100} total samples (train={num_train}, test>=100) "
            f"but only {X.shape[0]} available in {numbered_dataset_tag}"
        )
    train_indices, test_indices = get_train_test_indices(y, num_train, num_test, pos_ratio, seed)
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def get_xy_traintest(
    num_train,
    numbered_dataset_tag,
    layer,
    model_name=MODEL_NAME,
    MAX_AMT=5000,
    seed=42,
):
    return get_xy_traintest_specify(
        num_train, numbered_dataset_tag, layer, model_name,
        pos_ratio=0.5, MAX_AMT=MAX_AMT, seed=seed,
    )


def get_dataset_sizes():
    tags = get_numbered_binary_tags()
    return {tag: len(read_numbered_dataset_df(tag)) for tag in tags}


def get_training_sizes():
    # log-spaced training sizes from 2 to 1024
    return np.unique(
        np.round(np.logspace(1, 10, num=20, base=2)).astype(int)
    )


def get_class_imbalance():
    # 19 evenly spaced positive-class fractions from 5% to 95%
    return np.linspace(0.05, 0.95, num=19)


def get_classimabalance_num_train(numbered_dataset, min_num_test=100):
    # compute max feasible train/test sizes given the most extreme imbalance ratio
    y = get_yvals(numbered_dataset)
    points = get_class_imbalance()
    min_p, max_p = min(points), max(points)
    num_pos = np.sum(y)
    num_neg = len(y) - num_pos
    max_total = int(min(num_neg / (1 - min_p), num_pos / max_p))
    num_train = min(max_total - min_num_test, 1024)
    num_test = max(100, max_total - num_train - 1)
    return num_train, num_test


def corrupt_ytrain(ytrain, frac):
    # randomly flip `frac` fraction of training labels to simulate label noise
    assert 0 <= frac <= 0.5
    np.random.seed(42)
    num_to_flip = int(len(ytrain) * frac)
    flip_indices = np.random.choice(len(ytrain), size=num_to_flip, replace=False)
    ytrain_corrupted = ytrain.copy()
    ytrain_corrupted[flip_indices] = 1 - ytrain_corrupted[flip_indices]
    return ytrain_corrupted


def get_corrupt_frac():
    return np.linspace(0, 0.5, num=11)


def get_layers(model_name=MODEL_NAME):
    # detect available probe layers from saved activation filenames, then fall back to defaults
    known = {
        "gemma-2-9b":    ["embed", 9, 20, 31, 41],
        "llama-3.1-8b":  ["embed", 8, 16, 24, 31],
        "gemma-2-2b":    [12],
    }
    if model_name in known:
        return known[model_name]

    if model_name == "gemma-3-4b-it":
        files = glob.glob(f"data/model_activations_{model_name}/*.pt")
        layers: set[int] = set()
        for f in files:
            m = re.search(r"blocks\.(\d+)\.hook_resid_post\.pt", os.path.basename(f))
            if m:
                layers.add(int(m.group(1)))
        if layers:
            return ["embed"] + sorted(layers)
        # fallback: gemma-3-4b-it has 46 hidden layers, probe at ~21/48/74/98%
        fracs = [0.21, 0.48, 0.74, 0.98]
        computed = sorted({round(f * 45) for f in fracs})
        return ["embed"] + computed

    raise ValueError(f"Unsupported model_name: {model_name}")


def get_default_sae_layer(model_name=MODEL_NAME):
    # pick the middle probe layer, roughly 48% depth
    layers = [l for l in get_layers(model_name) if l != "embed"]
    return layers[len(layers) // 2]


def get_datasets(model_name=MODEL_NAME):
    # return datasets that have both activation files and entries in the master CSV
    act_dir = f"data/model_activations_{model_name}"
    if not os.path.exists(act_dir):
        return []
    dataset_sizes = get_dataset_sizes()
    datasets: set[str] = set()
    for fname in os.listdir(act_dir):
        if "blocks" not in fname:
            continue
        tag = fname.split("_blocks")[0]
        if tag in dataset_sizes:
            datasets.add(tag)
    return sorted(datasets)


def get_OOD_datasets(translation=True):
    paths = glob.glob("data/OOD data/*.csv")
    if translation:
        return [os.path.basename(p).replace("_OOD.csv", "") for p in paths]
    return [
        os.path.basename(p).replace("_OOD.csv", "")
        for p in paths
        if "translation" not in p
    ]


def get_xy_OOD(dataset, model_name=MODEL_NAME, layer=None):
    # load OOD activations and labels from the separate OOD activation directory
    if layer is None:
        layer = get_default_sae_layer(model_name)
    X = torch.load(
        f"data/model_activations_{model_name}_OOD/{dataset}_blocks.{layer}.hook_resid_post.pt",
        weights_only=False,
    )
    df = pd.read_csv(f"data/OOD data/{dataset}_OOD.csv")
    le = LabelEncoder()
    y = le.fit_transform(df["target"].values)
    return X, y


def get_OOD_traintest(dataset, model_name=MODEL_NAME, layer=None):
    # train on in-distribution data, test on OOD data from the shifted distribution
    if layer is None:
        layer = get_default_sae_layer(model_name)
    X_train, y_train, _, _ = get_xy_traintest_specify(
        num_train=1024,
        numbered_dataset_tag=dataset,
        layer=layer,
        model_name=model_name,
        MAX_AMT=1500,
        pos_ratio=0.5,
        num_test=0,
    )
    X_test, y_test = get_xy_OOD(dataset, model_name, layer)
    return X_train, y_train, X_test, y_test
