import os
import pickle
import random
import warnings

import torch
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from utils.data import (
    MODEL_NAME,
    SAE_ACT_BASEPATH,
    corrupt_ytrain,
    get_class_imbalance,
    get_corrupt_frac,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_OOD_datasets,
    get_training_sizes,
)
from utils.sae import get_sae_layers_gemma3, layer_to_sae_ids_gemma3
from utils.training import find_best_reg

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()
train_sizes = get_training_sizes()
corrupt_fracs = get_corrupt_frac()
fracs = get_class_imbalance()


def load_activations(path: str) -> torch.Tensor:
    # load a sparse-saved activation tensor and return it as a dense float
    return torch.load(path, weights_only=True).to_dense().float()


def description_string(dataset: str, layer: int, sae_id: str) -> str:
    # build filename stem from dataset, layer, and SAE id components
    width = sae_id.split("/")[1]
    k_str = sae_id.split("/")[2]
    return f"{dataset}_{layer}_{width}_{k_str}"


def get_sae_paths(
    dataset, layer, sae_id, reg_type, setting,
    binarize=False, num_train=None, corrupt_frac=None, frac=None,
):
    # resolve all file paths (save output, load activations) for a given probe configuration
    desc = description_string(dataset, layer, sae_id)

    if setting == "normal":
        extra = "_"
    elif setting == "scarcity":
        extra = f"_{num_train}_"
    elif setting == "label_noise":
        extra = f"_{corrupt_frac}_"
    elif setting == "class_imbalance":
        extra = f"_frac{frac}_"
    elif setting == "OOD":
        extra = "_"
    else:
        raise ValueError(f"Unknown setting: {setting}")

    reg_suffix = reg_type + ("_binarized" if binarize else "")

    save_setting = setting
    load_train_setting = "normal" if setting in ("label_noise", "OOD") else setting
    load_test_setting  = "OOD"    if setting == "OOD" else setting

    load_extra = "_" if setting == "label_noise" else extra

    save_path   = f"data/sae_probes_{MODEL_NAME}/{save_setting}_setting/{desc}{extra}{reg_suffix}.pkl"
    train_path  = f"{SAE_ACT_BASEPATH}/sae_activations_{MODEL_NAME}/{load_train_setting}_setting/{desc}{load_extra}X_train_sae.pt"
    test_path   = f"{SAE_ACT_BASEPATH}/sae_activations_{MODEL_NAME}/{load_test_setting}_setting/{desc}{load_extra}X_test_sae.pt"
    ytrain_path = f"{SAE_ACT_BASEPATH}/sae_activations_{MODEL_NAME}/{load_train_setting}_setting/{desc}{load_extra}y_train.pt"
    ytest_path  = f"{SAE_ACT_BASEPATH}/sae_activations_{MODEL_NAME}/{load_test_setting}_setting/{desc}{load_extra}y_test.pt"

    return {
        "save_path":    save_path,
        "train_path":   train_path,
        "test_path":    test_path,
        "y_train_path": ytrain_path,
        "y_test_path":  ytest_path,
    }


def get_sorted_indices(X_train_sae: torch.Tensor, y_train) -> torch.Tensor:
    # rank latents by normalised mean absolute difference between classes
    col_sums = X_train_sae.sum(dim=0)
    col_nonzero = (X_train_sae != 0).sum(dim=0)
    col_means = col_sums / (col_nonzero + 1e-6)
    X_norm = X_train_sae / (col_means + 1e-6)
    diff = X_norm[y_train == 1].mean(dim=0) - X_norm[y_train == 0].mean(dim=0)
    return torch.argsort(torch.abs(diff), descending=True)


def run_probe(
    dataset, layer, sae_id, reg_type, setting,
    binarize=False, num_train=None, corrupt_frac=None, frac=None,
) -> bool:
    # train and evaluate a sparse probe over top-k SAE latents and pickle the results
    paths = get_sae_paths(dataset, layer, sae_id, reg_type, setting,
                          binarize, num_train, corrupt_frac, frac)

    required = [paths["train_path"], paths["test_path"],
                paths["y_train_path"], paths["y_test_path"]]
    if not all(os.path.exists(p) for p in required):
        print(f"  Missing activation files for {dataset} layer={layer} SAE={sae_id}")
        for p in required:
            if not os.path.exists(p):
                print(f"    missing: {p}")
        return False

    X_train = load_activations(paths["train_path"])
    X_test  = load_activations(paths["test_path"])
    y_train = load_activations(paths["y_train_path"])
    y_test  = load_activations(paths["y_test_path"])

    if setting == "label_noise":
        y_train = corrupt_ytrain(y_train.numpy(), corrupt_frac)

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if setting == "normal" else [16, 128]
    sorted_idx = get_sorted_indices(X_train, y_train)

    all_metrics = []
    for k in tqdm(ks, desc=f"{dataset} L{layer} {sae_id}", leave=False):
        top_k = sorted_idx[:k]
        Xtr_k = X_train[:, top_k]
        Xte_k = X_test[:, top_k]

        if binarize and setting == "normal":
            Xtr_k = Xtr_k > 1
            Xte_k = Xte_k > 1

        metrics = find_best_reg(
            X_train=Xtr_k, y_train=y_train,
            X_test=Xte_k,  y_test=y_test,
            plot=False, n_jobs=-1, parallel=False,
            penalty=reg_type,
        )
        metrics.update({
            "k": k, "dataset": dataset, "layer": layer,
            "sae_id": sae_id, "reg_type": reg_type, "binarize": binarize,
        })
        if setting == "scarcity":
            metrics["num_train"] = num_train
        elif setting == "label_noise":
            metrics["corrupt_frac"] = corrupt_frac
        elif setting == "class_imbalance":
            metrics["frac"] = frac
        all_metrics.append(metrics)

    os.makedirs(os.path.dirname(paths["save_path"]), exist_ok=True)
    with open(paths["save_path"], "wb") as f:
        pickle.dump(all_metrics, f)
    print(f"  Saved → {paths['save_path']}")
    return True


def run_all_probes(
    reg_type: str,
    setting: str,
    binarize: bool = False,
    randomize_order: bool = False,
    dataset_filter: str | None = None,
):
    # iterate over all SAEs and datasets, running probes until all are complete
    all_layers = setting == "normal"
    layers = get_sae_layers_gemma3(all_layers=all_layers)

    loop_datasets = get_OOD_datasets() if setting == "OOD" else list(datasets)
    if dataset_filter:
        loop_datasets = [d for d in loop_datasets if d == dataset_filter]

    while True:
        found_missing = False

        if randomize_order:
            random.shuffle(loop_datasets)
            random.shuffle(layers)

        for dataset in loop_datasets:
            for layer in (random.sample(layers, len(layers)) if randomize_order else layers):
                sae_ids = layer_to_sae_ids_gemma3(layer)
                if randomize_order:
                    sae_ids = random.sample(sae_ids, len(sae_ids))

                for sae_id in sae_ids:
                    if setting == "normal":
                        p = get_sae_paths(dataset, layer, sae_id, reg_type, setting, binarize)
                        if not os.path.exists(p["save_path"]):
                            found_missing = True
                            run_probe(dataset, layer, sae_id, reg_type, setting, binarize)

                    elif setting == "scarcity":
                        for nt in train_sizes:
                            if nt > dataset_sizes[dataset] - 100:
                                continue
                            p = get_sae_paths(dataset, layer, sae_id, reg_type, setting,
                                              binarize, num_train=nt)
                            if not os.path.exists(p["save_path"]):
                                found_missing = True
                                run_probe(dataset, layer, sae_id, reg_type, setting,
                                          binarize, num_train=nt)

                    elif setting == "label_noise":
                        for cf in corrupt_fracs:
                            p = get_sae_paths(dataset, layer, sae_id, reg_type, setting,
                                              binarize, corrupt_frac=cf)
                            if not os.path.exists(p["save_path"]):
                                found_missing = True
                                run_probe(dataset, layer, sae_id, reg_type, setting,
                                          binarize, corrupt_frac=cf)

                    elif setting == "class_imbalance":
                        for fr in fracs:
                            p = get_sae_paths(dataset, layer, sae_id, reg_type, setting,
                                              binarize, frac=fr)
                            if not os.path.exists(p["save_path"]):
                                found_missing = True
                                run_probe(dataset, layer, sae_id, reg_type, setting,
                                          binarize, frac=fr)

                    elif setting == "OOD":
                        p = get_sae_paths(dataset, layer, sae_id, reg_type, setting, binarize)
                        if not os.path.exists(p["save_path"]):
                            found_missing = True
                            run_probe(dataset, layer, sae_id, reg_type, setting, binarize)

        if not found_missing:
            print(f"All {setting} probes complete.")
            break


if __name__ == "__main__":
    REG_TYPE = "l1"       # one of: l1, l2
    SETTING = "normal"    # one of: normal, scarcity, label_noise, class_imbalance, OOD
    BINARIZE = False
    RANDOMIZE_ORDER = False
    DATASET = None        # set to a specific dataset tag to only process that one

    run_all_probes(REG_TYPE, SETTING, BINARIZE, RANDOMIZE_ORDER, DATASET)
