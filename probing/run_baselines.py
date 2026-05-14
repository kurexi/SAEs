import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data import (
    MODEL_NAME,
    corrupt_ytrain,
    get_class_imbalance,
    get_classimabalance_num_train,
    get_corrupt_frac,
    get_dataset_sizes,
    get_datasets,
    get_layers,
    get_numbered_binary_tags,
    get_OOD_datasets,
    get_OOD_traintest,
    get_training_sizes,
    get_xy_traintest,
    get_xy_traintest_specify,
    get_default_sae_layer,
)
from utils.training import (
    find_best_knn,
    find_best_mlp,
    find_best_pcareg,
    find_best_reg,
    find_best_xgboost,
)

dataset_sizes = get_dataset_sizes()
datasets_all  = get_numbered_binary_tags()

METHODS = {
    "logreg":  find_best_reg,
    "pca":     find_best_pcareg,
    "knn":     find_best_knn,
    "xgboost": find_best_xgboost,
    "mlp":     find_best_mlp,
}


def run_baseline_normal(layer, dataset, method_name):
    # run a single baseline method on one dataset/layer and save the result
    savepath = (
        f"data/baseline_results_{MODEL_NAME}/normal/allruns/"
        f"layer{layer}_{dataset}_{method_name}.csv"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    num_train = min(dataset_sizes[dataset] - 100, 1024)
    try:
        X_train, y_train, X_test, y_test = get_xy_traintest(
            num_train, dataset, layer, MODEL_NAME
        )
    except Exception as e:
        print(f"  skip {dataset} layer={layer}: {e}")
        return None
    metrics = METHODS[method_name](X_train, y_train, X_test, y_test)
    row = {"dataset": dataset, "method": method_name, **metrics}
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_normal(dataset_filter=None):
    avail = list(get_datasets(MODEL_NAME))
    if dataset_filter:
        avail = [d for d in avail if d == dataset_filter]
    np.random.shuffle(avail)
    for method in METHODS:
        for layer in get_layers(MODEL_NAME):
            for dataset in avail:
                run_baseline_normal(layer, dataset, method)


def coalesce_normal():
    # aggregate per-run CSVs into one summary CSV per layer
    for layer in get_layers(MODEL_NAME):
        results = []
        for dataset in datasets_all:
            for method in METHODS:
                p = (
                    f"data/baseline_results_{MODEL_NAME}/normal/allruns/"
                    f"layer{layer}_{dataset}_{method}.csv"
                )
                if os.path.exists(p):
                    results.append(pd.read_csv(p))
                else:
                    print(f"  missing: layer{layer} {dataset} {method}")
        if results:
            out = f"results/baseline_probes_{MODEL_NAME}/normal_settings/layer{layer}_results.csv"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pd.concat(results, ignore_index=True).to_csv(out, index=False)
            print(f"  → {out}")


def run_baseline_scarcity(num_train, dataset, method_name, layer):
    # run a baseline at a specific training size for the data-scarcity setting
    savepath = (
        f"data/baseline_results_{MODEL_NAME}/scarcity/allruns/"
        f"layer{layer}_{dataset}_{method_name}_numtrain{num_train}.csv"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    if num_train > dataset_sizes[dataset] - 100:
        return None
    try:
        X_train, y_train, X_test, y_test = get_xy_traintest(
            num_train, dataset, layer, MODEL_NAME
        )
    except Exception as e:
        print(f"  skip {dataset} n={num_train}: {e}")
        return None
    metrics = METHODS[method_name](X_train, y_train, X_test, y_test)
    row = {"dataset": dataset, "method": method_name, "num_train": num_train, **metrics}
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_scarcity(layer, dataset_filter=None):
    avail = list(get_datasets(MODEL_NAME))
    if dataset_filter:
        avail = [d for d in avail if d == dataset_filter]
    np.random.shuffle(avail)
    for method in METHODS:
        for nt in get_training_sizes():
            for dataset in avail:
                run_baseline_scarcity(nt, dataset, method, layer)


def coalesce_scarcity(layer):
    results = []
    outdir = f"results/baseline_probes_{MODEL_NAME}/scarcity"
    os.makedirs(outdir, exist_ok=True)
    for dataset in datasets_all:
        for nt in get_training_sizes():
            for method in METHODS:
                p = (
                    f"data/baseline_results_{MODEL_NAME}/scarcity/allruns/"
                    f"layer{layer}_{dataset}_{method}_numtrain{nt}.csv"
                )
                if os.path.exists(p):
                    results.append(pd.read_csv(p))
    if results:
        out = f"{outdir}/all_results.csv"
        pd.concat(results, ignore_index=True).to_csv(out, index=False)
        print(f"  → {out}")


def run_baseline_imbalance(frac, dataset, method_name, layer):
    # run a baseline at a specific class-imbalance ratio
    frac_r = round(frac * 20) / 20
    savepath = (
        f"data/baseline_results_{MODEL_NAME}/imbalance/allruns/"
        f"layer{layer}_{dataset}_{method_name}_frac{frac_r}.csv"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    num_train, num_test = get_classimabalance_num_train(dataset)
    try:
        X_train, y_train, X_test, y_test = get_xy_traintest_specify(
            num_train, dataset, layer, MODEL_NAME,
            pos_ratio=frac_r, num_test=num_test,
        )
    except Exception as e:
        print(f"  skip {dataset} frac={frac_r}: {e}")
        return None
    metrics = METHODS[method_name](X_train, y_train, X_test, y_test)
    row = {"dataset": dataset, "method": method_name,
           "ratio": frac_r, "num_train": num_train, **metrics}
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_imbalance(layer, dataset_filter=None):
    avail = list(get_datasets(MODEL_NAME))
    if dataset_filter:
        avail = [d for d in avail if d == dataset_filter]
    np.random.shuffle(avail)
    for method in METHODS:
        for frac in get_class_imbalance():
            for dataset in avail:
                run_baseline_imbalance(frac, dataset, method, layer)


def coalesce_imbalance(layer):
    results = []
    outdir = f"results/baseline_probes_{MODEL_NAME}/imbalance"
    os.makedirs(outdir, exist_ok=True)
    for dataset in datasets_all:
        for frac in get_class_imbalance():
            for method in METHODS:
                frac_r = round(frac * 20) / 20
                p = (
                    f"data/baseline_results_{MODEL_NAME}/imbalance/allruns/"
                    f"layer{layer}_{dataset}_{method}_frac{frac_r}.csv"
                )
                if os.path.exists(p):
                    results.append(pd.read_csv(p))
    if results:
        out = f"{outdir}/all_results.csv"
        pd.concat(results, ignore_index=True).to_csv(out, index=False)
        print(f"  → {out}")


def run_baseline_corrupt(frac, dataset, method_name, layer):
    # run a baseline with corrupted training labels (label noise setting)
    frac_r = round(frac * 20) / 20
    savepath = (
        f"data/baseline_results_{MODEL_NAME}/corrupt/allruns/"
        f"layer{layer}_{dataset}_{method_name}_corrupt{frac_r}.csv"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if os.path.exists(savepath):
        return None
    num_train = min(dataset_sizes[dataset] - 100, 1024)
    try:
        X_train, y_train, X_test, y_test = get_xy_traintest(
            num_train, dataset, layer, MODEL_NAME
        )
    except Exception as e:
        print(f"  skip {dataset} frac={frac_r}: {e}")
        return None
    y_train = corrupt_ytrain(y_train, frac_r)
    metrics = METHODS[method_name](X_train, y_train, X_test, y_test)
    row = {"dataset": dataset, "method": method_name,
           "ratio": frac_r, "num_train": num_train, **metrics}
    pd.DataFrame([row]).to_csv(savepath, index=False)
    return True


def run_all_corrupt(layer, dataset_filter=None):
    avail = list(get_datasets(MODEL_NAME))
    if dataset_filter:
        avail = [d for d in avail if d == dataset_filter]
    np.random.shuffle(avail)
    for frac in get_corrupt_frac():
        for dataset in avail:
            run_baseline_corrupt(frac, dataset, "logreg", layer)


def coalesce_corrupt(layer):
    results = []
    outdir = f"results/baseline_probes_{MODEL_NAME}/corrupt"
    os.makedirs(outdir, exist_ok=True)
    for dataset in datasets_all:
        for frac in get_corrupt_frac():
            frac_r = round(frac * 20) / 20
            p = (
                f"data/baseline_results_{MODEL_NAME}/corrupt/allruns/"
                f"layer{layer}_{dataset}_logreg_corrupt{frac_r}.csv"
            )
            if os.path.exists(p):
                results.append(pd.read_csv(p))
    if results:
        out = f"{outdir}/all_results.csv"
        pd.concat(results, ignore_index=True).to_csv(out, index=False)
        print(f"  → {out}")


def run_all_ood(layer, dataset_filter=None):
    # run logistic regression on OOD test sets using ID training data
    allruns_dir = f"data/baseline_results_{MODEL_NAME}/ood/allruns"
    os.makedirs(allruns_dir, exist_ok=True)
    ood_datasets = get_OOD_datasets()
    if dataset_filter:
        ood_datasets = [d for d in ood_datasets if d == dataset_filter]
    for dataset in tqdm(ood_datasets, desc="OOD"):
        savepath = os.path.join(allruns_dir, f"{dataset}.csv")
        if os.path.exists(savepath):
            continue
        try:
            X_train, y_train, X_test, y_test = get_OOD_traintest(
                dataset, MODEL_NAME, layer
            )
        except Exception as e:
            print(f"  skip {dataset}: {e}")
            continue
        metrics = find_best_reg(X_train, y_train, X_test, y_test, penalty="l2")
        pd.DataFrame([{"dataset": dataset, "test_auc_baseline": metrics["test_auc"]}]).to_csv(
            savepath, index=False
        )
        print(f"  → {savepath}")


def coalesce_ood():
    allruns_dir = f"data/baseline_results_{MODEL_NAME}/ood/allruns"
    outdir = f"results/baseline_probes_{MODEL_NAME}/ood"
    os.makedirs(outdir, exist_ok=True)
    results = []
    for dataset in get_OOD_datasets():
        p = os.path.join(allruns_dir, f"{dataset}.csv")
        if os.path.exists(p):
            results.append(pd.read_csv(p))
        else:
            print(f"  missing: {dataset}")
    if results:
        out = f"{outdir}/all_results.csv"
        pd.concat(results, ignore_index=True).to_csv(out, index=False)
        print(f"  → {out}")


if __name__ == "__main__":
    CONDITION = "normal"   # one of: normal, scarcity, imbalance, corrupt, OOD
    LAYER = None           # set to an int to override; defaults to the middle probe layer
    COALESCE = False       # set to True to coalesce per-run files into a summary CSV
    DATASET = None         # set to a specific dataset tag to only process that one

    layer = LAYER if LAYER is not None else get_default_sae_layer(MODEL_NAME)

    if COALESCE:
        if CONDITION == "normal":
            coalesce_normal()
        elif CONDITION == "scarcity":
            coalesce_scarcity(layer)
        elif CONDITION == "imbalance":
            coalesce_imbalance(layer)
        elif CONDITION == "corrupt":
            coalesce_corrupt(layer)
        elif CONDITION == "OOD":
            coalesce_ood()
    else:
        if CONDITION == "normal":
            run_all_normal(DATASET)
        elif CONDITION == "scarcity":
            run_all_scarcity(layer, DATASET)
        elif CONDITION == "imbalance":
            run_all_imbalance(layer, DATASET)
        elif CONDITION == "corrupt":
            run_all_corrupt(layer, DATASET)
        elif CONDITION == "OOD":
            run_all_ood(layer, DATASET)
