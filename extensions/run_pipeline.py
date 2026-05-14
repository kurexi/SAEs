import os
import pickle

import torch
from tqdm import tqdm
import pandas as pd

from probing.run_baselines import coalesce_imbalance, coalesce_normal, coalesce_scarcity
from probing.train_probes import description_string, get_sorted_indices
from setup.generate_sae_activations import (
    _encode_batched,
    imbalance_paths,
    load_activations,
    normal_paths,
    ood_paths,
    save_activations,
    scarcity_paths,
)
from utils.sae import get_sae_layers_gemma3, layer_to_sae_ids_gemma3, sae_id_to_sae_gemma3
from utils.data import (
    MODEL_NAME,
    SAE_ACT_BASEPATH,
    corrupt_ytrain,
    get_class_imbalance,
    get_classimabalance_num_train,
    get_dataset_sizes,
    get_default_sae_layer,
    get_numbered_binary_tags,
    get_OOD_datasets,
    get_OOD_traintest,
    get_training_sizes,
    get_xy_traintest,
    get_xy_traintest_specify,
)
from utils.training import find_best_reg

dataset_sizes = get_dataset_sizes()

METHODS = {
    "logreg": find_best_reg,
}


def run_probe(
    train: str, test: str, y_train: str, y_test: str,
    dataset, layer, sae_id, reg_type, setting,
    binarize=False, num_train=None, corrupt_frac=None, frac=None,
) -> bool:
    # load pre-saved SAE activations and run a sparse linear probe over top-k latents
    required = [train, test, y_train, y_test]
    if not all(os.path.exists(p) for p in required):
        print(f"  Missing activation files for {dataset} layer={layer} SAE={sae_id}")
        for p in required:
            if not os.path.exists(p):
                print(f"    missing: {p}")
        return False

    X_train = load_activations(train)
    X_test  = load_activations(test)
    y_train = load_activations(y_train)
    y_test  = load_activations(y_test)

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
        elif setting == "imbalance":
            metrics["frac"] = frac
        all_metrics.append(metrics)

    if setting == "normal":
        extra = "_"
    elif setting == "scarcity":
        extra = f"_{num_train}_"
    elif setting == "label_noise":
        extra = f"_{corrupt_frac}_"
    elif setting == "imbalance":
        extra = f"_frac{frac}_"
    elif setting == "OOD":
        extra = "_"
    else:
        extra = ""
    desc = description_string(dataset, layer, sae_id)
    reg_suffix = reg_type + ("_binarized" if binarize else "")
    save_path = f"data/sae_probes_{MODEL_NAME}/{setting}_setting/{desc}{extra}{reg_suffix}.pkl"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(all_metrics, f)
    print(f"  Saved → {save_path}")
    return True


def gen_activations_and_train_baselines(setting: str, dataset: str, device: str):
    # generate SAE activations on the fly and immediately train probes and baselines
    all_layers = setting == "normal"
    layers = get_sae_layers_gemma3(all_layers=all_layers)

    for layer in layers:
        sae_ids = layer_to_sae_ids_gemma3(layer)
        for sae_id in sae_ids:
            print(f"  SAE {sae_id}  layer={layer}  setting={setting}")
            try:
                sae = sae_id_to_sae_gemma3(sae_id, device)
            except (FileNotFoundError, ValueError) as e:
                print(f"    skipping {sae_id}: {e}")
                continue

            if setting == "normal":
                p = normal_paths(dataset, layer, sae_id)
                if all(os.path.exists(v) for v in p.values()):
                    continue

                size = dataset_sizes[dataset]
                num_train = min(size - 100, 1024)
                try:
                    X_train, y_train, X_test, y_test = get_xy_traintest(
                        num_train, dataset, layer, MODEL_NAME
                    )
                except Exception as e:
                    print(f"    skip {dataset}: {e}")
                    continue

                print(f"    Generating activations and training probes for {dataset} layer={layer} SAE={sae_id} …")

                save_activations(p["train"],  _encode_batched(sae, X_train, device))
                save_activations(p["test"],   _encode_batched(sae, X_test,  device))
                save_activations(p["ytrain"], torch.tensor(y_train))
                save_activations(p["ytest"],  torch.tensor(y_test))
                run_probe(
                    p["train"], p["test"], p["ytrain"], p["ytest"],
                    dataset, layer, sae_id, "l1", setting, False)

                for method_name in METHODS.keys():
                    savepath = (
                        f"data/baseline_results_{MODEL_NAME}/normal/allruns/"
                        f"layer{layer}_{dataset}_{method_name}.csv"
                    )
                    os.makedirs(os.path.dirname(savepath), exist_ok=True)
                    if os.path.exists(savepath):
                        continue
                    metrics = METHODS[method_name](X_train, y_train, X_test, y_test)
                    row = {"dataset": dataset, "method": method_name, **metrics}
                    pd.DataFrame([row]).to_csv(savepath, index=False)

                os.remove(p["train"])
                os.remove(p["test"])
                os.remove(p["ytrain"])
                os.remove(p["ytest"])

            elif setting == "scarcity":
                for num_train in get_training_sizes():
                    if num_train > dataset_sizes[dataset] - 100:
                        continue
                    p = scarcity_paths(dataset, layer, sae_id, num_train)

                    if all(os.path.exists(v) for v in p.values()):
                        continue

                    try:
                        X_train, y_train, X_test, y_test = get_xy_traintest(
                            num_train, dataset, layer, MODEL_NAME
                        )
                    except Exception as e:
                        print(f"    skip {dataset}: {e}")
                        continue

                    print(f"    Generating activations and training probes for {dataset} layer={layer} SAE={sae_id} …")

                    save_activations(p["train"],  _encode_batched(sae, X_train, device))
                    save_activations(p["test"],   _encode_batched(sae, X_test,  device))
                    save_activations(p["ytrain"], torch.tensor(y_train))
                    save_activations(p["ytest"],  torch.tensor(y_test))
                    run_probe(
                        p["train"], p["test"], p["ytrain"], p["ytest"],
                        dataset, layer, sae_id, "l1", setting, False, num_train)

                    for method_name in METHODS.keys():
                        savepath = (
                            f"data/baseline_results_{MODEL_NAME}/scarcity/allruns/"
                            f"layer{layer}_{dataset}_{method_name}_numtrain{num_train}.csv"
                        )
                        os.makedirs(os.path.dirname(savepath), exist_ok=True)
                        if os.path.exists(savepath):
                            continue
                        metrics = METHODS[method_name](X_train, y_train, X_test, y_test)
                        row = {"dataset": dataset, "method": method_name, **metrics}
                        pd.DataFrame([row]).to_csv(savepath, index=False)

                    os.remove(p["train"])
                    os.remove(p["test"])
                    os.remove(p["ytrain"])
                    os.remove(p["ytest"])

            elif setting == "imbalance":
                for frac in get_class_imbalance():
                    p = imbalance_paths(dataset, layer, sae_id, frac)
                    if all(os.path.exists(v) for v in p.values()):
                        continue
                    num_train, num_test = get_classimabalance_num_train(dataset)
                    try:
                        X_train, y_train, X_test, y_test = get_xy_traintest_specify(
                            num_train, dataset, layer, MODEL_NAME,
                            pos_ratio=frac, num_test=num_test,
                        )
                    except Exception as e:
                        print(f"    skip {dataset} frac={frac:.2f}: {e}")
                        continue
                    save_activations(p["train"],  _encode_batched(sae, X_train, device))
                    save_activations(p["test"],   _encode_batched(sae, X_test,  device))
                    save_activations(p["ytrain"], torch.tensor(y_train))
                    save_activations(p["ytest"],  torch.tensor(y_test))
                    run_probe(
                        p["train"], p["test"], p["ytrain"], p["ytest"],
                        dataset, layer, sae_id, "l1", setting, False, frac=frac)

                    print(f"    Generating activations and training probes for {dataset} layer={layer} SAE={sae_id} …")

                    for method_name in METHODS.keys():
                        savepath = (
                            f"data/baseline_results_{MODEL_NAME}/imbalance/allruns/"
                            f"layer{layer}_{dataset}_{method_name}_frac{frac:.2f}.csv"
                        )
                        os.makedirs(os.path.dirname(savepath), exist_ok=True)
                        if os.path.exists(savepath):
                            continue
                        metrics = METHODS[method_name](X_train, y_train, X_test, y_test)
                        row = {"dataset": dataset, "method": method_name, **metrics}
                        pd.DataFrame([row]).to_csv(savepath, index=False)

                    os.remove(p["train"])
                    os.remove(p["test"])
                    os.remove(p["ytrain"])
                    os.remove(p["ytest"])


def run_main_dataset(dataset: str, device: str):
    # run scarcity and imbalance settings end-to-end for one dataset
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*60}")

    for setting in ["scarcity", "imbalance"]:
        print(f"    setting={setting}")
        gen_activations_and_train_baselines(setting, dataset, device)


def run_ood_dataset(dataset: str, device: str, layer: int):
    # generate OOD activations, train probes, and save baseline results for one dataset
    print(f"\n{'='*60}")
    print(f"  OOD Dataset: {dataset}")
    print(f"{'='*60}")

    sae_ids = layer_to_sae_ids_gemma3(layer)
    for sae_id in sae_ids:
        print(f"  SAE {sae_id}  layer={layer}  setting=OOD")
        try:
            sae = sae_id_to_sae_gemma3(sae_id, device)
        except (FileNotFoundError, ValueError) as e:
            print(f"    skipping {sae_id}: {e}")
            continue

        p = ood_paths(dataset, layer, sae_id)
        if all(os.path.exists(v) for v in p.values()):
            continue

        try:
            X_train, y_train, X_test, y_test = get_OOD_traintest(dataset, MODEL_NAME, layer)
        except Exception as e:
            print(f"    skip {dataset}: {e}")
            continue

        print(f"    Generating activations and training probes for {dataset} layer={layer} SAE={sae_id} …")
        save_activations(p["train"],  _encode_batched(sae, X_train, device))
        save_activations(p["test"],   _encode_batched(sae, X_test,  device))
        save_activations(p["ytrain"], torch.tensor(y_train))
        save_activations(p["ytest"],  torch.tensor(y_test))
        run_probe(
            p["train"], p["test"], p["ytrain"], p["ytest"],
            dataset, layer, sae_id, "l1", "OOD", False)

        for method_name in METHODS.keys():
            savepath = (
                f"data/baseline_results_{MODEL_NAME}/ood/allruns/"
                f"{dataset}.csv"
            )
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            if os.path.exists(savepath):
                continue
            result = METHODS[method_name](X_train, y_train, X_test, y_test)
            metrics = result[0] if isinstance(result, tuple) else result
            pd.DataFrame([{"dataset": dataset, "test_auc_baseline": metrics["test_auc"]}]).to_csv(
                savepath, index=False
            )

        os.remove(p["train"])
        os.remove(p["test"])
        os.remove(p["ytrain"])
        os.remove(p["ytest"])


if __name__ == "__main__":
    DEVICE = "cuda"
    RUN_MAIN = False   # set to True to run normal/scarcity/imbalance settings
    RUN_OOD = True     # set to True to run OOD datasets

    layer = get_default_sae_layer(MODEL_NAME)
    if not isinstance(layer, int):
        raise ValueError(f"Expected integer SAE layer, got: {layer}")
    print(f"Default SAE layer: {layer}  |  device: {DEVICE}")

    if RUN_MAIN:
        main_datasets = get_numbered_binary_tags()
        print(f"\nProcessing {len(main_datasets)} main datasets …")
        for dataset in main_datasets:
            run_main_dataset(dataset, DEVICE)

    if RUN_OOD:
        ood_datasets = get_OOD_datasets()
        print(f"\nProcessing {len(ood_datasets)} OOD datasets …")
        for dataset in ood_datasets:
            run_ood_dataset(dataset, DEVICE, layer)
