import os
import random

import torch
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from utils.data import (
    MODEL_NAME,
    SAE_ACT_BASEPATH,
    get_dataset_sizes,
    get_numbered_binary_tags,
    get_OOD_datasets,
    get_classimabalance_num_train,
    get_class_imbalance,
    get_training_sizes,
    get_xy_OOD,
    get_xy_traintest,
    get_xy_traintest_specify,
)
from utils.sae import (
    layer_to_sae_ids_gemma3,
    sae_id_to_sae_gemma3,
    get_sae_layers_gemma3,
    DEFAULT_SAE_K,
)

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()


def save_activations(path: str, activation: torch.Tensor):
    # save sparse to cut storage (SAE outputs are typically very sparse)
    torch.save(activation.to_sparse(), path)


def load_activations(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True).to_dense().float()


def _encode_batched(sae, X: torch.Tensor, device: str, batch_size: int = 128) -> torch.Tensor:
    # encode a tensor through the SAE in chunks to avoid OOM
    parts = []
    for i in range(0, len(X), batch_size):
        parts.append(sae.encode(X[i: i + batch_size].to(device)).cpu())
    return torch.cat(parts)


def description_string(dataset: str, layer: int, sae_id: str) -> str:
    # build the filename stem, e.g. "1_dataset_22_width_16k_topk_32"
    width = sae_id.split("/")[1]
    sae_type = sae_id.split("/")[2]
    return f"{dataset}_{layer}_{width}_{sae_type}"


def normal_paths(dataset, layer, sae_id):
    d = f"{SAE_ACT_BASEPATH}/sae_activations_{MODEL_NAME}/normal_setting"
    os.makedirs(d, exist_ok=True)
    desc = description_string(dataset, layer, sae_id)
    return {
        "train":  os.path.join(d, f"{desc}_X_train_sae.pt"),
        "test":   os.path.join(d, f"{desc}_X_test_sae.pt"),
        "ytrain": os.path.join(d, f"{desc}_y_train.pt"),
        "ytest":  os.path.join(d, f"{desc}_y_test.pt"),
    }


def scarcity_paths(dataset, layer, sae_id, num_train):
    d = f"{SAE_ACT_BASEPATH}/sae_activations_{MODEL_NAME}/scarcity_setting"
    os.makedirs(d, exist_ok=True)
    desc = description_string(dataset, layer, sae_id)
    return {
        "train":  os.path.join(d, f"{desc}_{num_train}_X_train_sae.pt"),
        "test":   os.path.join(d, f"{desc}_{num_train}_X_test_sae.pt"),
        "ytrain": os.path.join(d, f"{desc}_{num_train}_y_train.pt"),
        "ytest":  os.path.join(d, f"{desc}_{num_train}_y_test.pt"),
    }


def imbalance_paths(dataset, layer, sae_id, frac):
    d = f"{SAE_ACT_BASEPATH}/sae_activations_{MODEL_NAME}/class_imbalance_setting"
    os.makedirs(d, exist_ok=True)
    desc = description_string(dataset, layer, sae_id)
    return {
        "train":  os.path.join(d, f"{desc}_frac{frac}_X_train_sae.pt"),
        "test":   os.path.join(d, f"{desc}_frac{frac}_X_test_sae.pt"),
        "ytrain": os.path.join(d, f"{desc}_frac{frac}_y_train.pt"),
        "ytest":  os.path.join(d, f"{desc}_frac{frac}_y_test.pt"),
    }


def ood_paths(dataset, layer, sae_id):
    norm_d = f"{SAE_ACT_BASEPATH}/sae_activations_{MODEL_NAME}/normal_setting"
    ood_d  = f"{SAE_ACT_BASEPATH}/sae_activations_{MODEL_NAME}/OOD_setting"
    os.makedirs(ood_d, exist_ok=True)
    desc = description_string(dataset, layer, sae_id)
    return {
        # train data is shared with normal_setting to avoid duplication
        "train":  os.path.join(norm_d, f"{desc}_X_train_sae.pt"),
        "test":   os.path.join(ood_d,  f"{desc}_X_test_sae.pt"),
        "ytrain": os.path.join(norm_d, f"{desc}_y_train.pt"),
        "ytest":  os.path.join(ood_d,  f"{desc}_y_test.pt"),
    }


def run_normal(layer, sae, sae_id, device, dataset_filter=None):
    # encode and save train/test SAE activations for the normal setting
    _datasets = [dataset_filter] if dataset_filter else datasets
    for dataset in _datasets:
        p = normal_paths(dataset, layer, sae_id)
        if all(os.path.exists(v) for v in p.values()):
            continue
        size = dataset_sizes[dataset]
        num_train = min(size - 100, 1024)
        try:
            X_train, y_train, X_test, y_test = get_xy_traintest(num_train, dataset, layer, MODEL_NAME)
        except Exception as e:
            print(f"    skip {dataset}: {e}")
            continue
        save_activations(p["train"],  _encode_batched(sae, X_train, device))
        save_activations(p["test"],   _encode_batched(sae, X_test,  device))
        save_activations(p["ytrain"], torch.tensor(y_train))
        save_activations(p["ytest"],  torch.tensor(y_test))


def run_scarcity(layer, sae, sae_id, device, dataset_filter=None):
    # encode and save SAE activations across all training-size checkpoints
    _datasets = [dataset_filter] if dataset_filter else datasets
    for dataset in _datasets:
        for num_train in get_training_sizes():
            if num_train > dataset_sizes[dataset] - 100:
                continue
            p = scarcity_paths(dataset, layer, sae_id, num_train)
            if all(os.path.exists(v) for v in p.values()):
                continue
            try:
                X_train, y_train, X_test, y_test = get_xy_traintest(num_train, dataset, layer, MODEL_NAME)
            except Exception as e:
                print(f"    skip {dataset} n={num_train}: {e}")
                continue
            save_activations(p["train"],  _encode_batched(sae, X_train, device))
            save_activations(p["test"],   _encode_batched(sae, X_test,  device))
            save_activations(p["ytrain"], torch.tensor(y_train))
            save_activations(p["ytest"],  torch.tensor(y_test))


def run_imbalance(layer, sae, sae_id, device, dataset_filter=None):
    # encode and save SAE activations across all positive-class fraction checkpoints
    _datasets = [dataset_filter] if dataset_filter else datasets
    for dataset in _datasets:
        for frac in get_class_imbalance():
            p = imbalance_paths(dataset, layer, sae_id, frac)
            if all(os.path.exists(v) for v in p.values()):
                continue
            num_train, num_test = get_classimabalance_num_train(dataset)
            try:
                X_train, y_train, X_test, y_test = get_xy_traintest_specify(
                    num_train, dataset, layer, MODEL_NAME, pos_ratio=frac, num_test=num_test,
                )
            except Exception as e:
                print(f"    skip {dataset} frac={frac:.2f}: {e}")
                continue
            save_activations(p["train"],  _encode_batched(sae, X_train, device))
            save_activations(p["test"],   _encode_batched(sae, X_test,  device))
            save_activations(p["ytrain"], torch.tensor(y_train))
            save_activations(p["ytest"],  torch.tensor(y_test))


def run_ood(layer, sae, sae_id, device, dataset_filter=None):
    # encode in-distribution train and OOD test activations
    _datasets = [dataset_filter] if dataset_filter else get_OOD_datasets()
    for dataset in _datasets:
        p = ood_paths(dataset, layer, sae_id)

        if not os.path.exists(p["train"]):
            size = dataset_sizes.get(dataset, 0)
            num_train = min(max(size - 100, 0), 1024)
            if num_train < 2:
                print(f"    skip OOD train {dataset}: not enough data")
                continue
            try:
                X_train, y_train, _, _ = get_xy_traintest(num_train, dataset, layer, MODEL_NAME)
            except Exception as e:
                print(f"    skip {dataset} OOD train: {e}")
                continue
            save_activations(p["train"],  _encode_batched(sae, X_train, device))
            save_activations(p["ytrain"], torch.tensor(y_train))

        if not os.path.exists(p["test"]):
            try:
                X_test, y_test = get_xy_OOD(dataset, MODEL_NAME, layer)
            except Exception as e:
                print(f"    skip {dataset} OOD test: {e}")
                continue
            save_activations(p["test"],  _encode_batched(sae, X_test, device))
            save_activations(p["ytest"], torch.tensor(y_test))


WORKERS = {
    "normal":    run_normal,
    "scarcity":  run_scarcity,
    "imbalance": run_imbalance,
    "OOD":       run_ood,
}


def process_setting(setting: str, device: str, randomize_order: bool = False, dataset_filter: str | None = None):
    # iterate over all SAEs and encode activations for every dataset under the given setting
    all_layers = setting == "normal"
    layers = get_sae_layers_gemma3(all_layers=all_layers)

    if randomize_order:
        random.shuffle(layers)

    for layer in layers:
        sae_ids = layer_to_sae_ids_gemma3(layer)
        if randomize_order:
            random.shuffle(sae_ids)

        for sae_id in sae_ids:
            print(f"  SAE {sae_id}  layer={layer}  setting={setting}")
            try:
                sae = sae_id_to_sae_gemma3(sae_id, device)
            except (FileNotFoundError, ValueError) as e:
                print(f"    skipping {sae_id}: {e}")
                continue
            WORKERS[setting](layer, sae, sae_id, device, dataset_filter)

    print(f"Done: {setting} setting.")


if __name__ == "__main__":
    DEVICE = "cuda:0"
    SETTING = "normal"  # one of: normal, scarcity, imbalance, OOD
    DATASET = None      # set to a specific dataset tag to only process that one

    process_setting(SETTING, DEVICE, dataset_filter=DATASET)
