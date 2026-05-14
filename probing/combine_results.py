import glob
import os
import pickle

import pandas as pd
from tqdm import tqdm

MODEL_NAME = "gemma-3-4b-it"
SETTINGS = ["normal", "scarcity", "class_imbalance", "label_noise", "OOD"]


def load_pickle(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception as e:
            print(f"  Bad file {path}: {e}")
            return None


def process_setting(setting: str):
    # collect all per-probe pickle files for a setting and write a single combined CSV
    in_pattern = f"data/sae_probes_{MODEL_NAME}/{setting}_setting/*.pkl"
    out_dir     = f"results/sae_probes_{MODEL_NAME}/{setting}_setting"

    files = glob.glob(in_pattern)
    if not files:
        print(f"  [{setting}] No files found — skipping.")
        return

    print(f"  [{setting}] Found {len(files)} files …")
    all_metrics = []
    bad = []

    for f in tqdm(files, desc=setting, leave=False):
        metrics = load_pickle(f)
        if metrics:
            all_metrics.extend(metrics)
        else:
            bad.append(f)

    if bad:
        print(f"  [{setting}] WARNING: {len(bad)} unreadable files:")
        for b in bad:
            print(f"    {b}")

    if not all_metrics:
        print(f"  [{setting}] No valid metrics — skipping CSV write.")
        return

    df = pd.DataFrame(all_metrics)
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/all_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"  [{setting}] {len(df)} rows → {out_path}")


if __name__ == "__main__":
    SETTING = None   # set to one of SETTINGS to process only that one, or None for all

    settings = [SETTING] if SETTING else SETTINGS
    for s in settings:
        process_setting(s)
