#!/bin/bash -l
#SBATCH -p ecsstudents_l4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=20:00:00
#SBATCH -o logs/pipeline_setup_%j.out
#SBATCH -e logs/pipeline_setup_%j.err
#SBATCH -J pipeline_setup

# Steps 1-7: download -> activations -> train SAE -> SAE activations ->
#            baselines -> probes (all 5 settings) -> aggregate results
#
# NOTE: SAE training alone can take the full 20h. If the cluster allows a
# longer wall time, increase --time accordingly.

module load conda/python3
conda init
source ~/.bashrc
conda activate SAE

export HF_TOKEN=hf_PUT_TOKEN_HERE

mkdir -p logs
cd SAE

# --- Step 1: download model weights (needs internet, offline flags OFF) ---
python setup/download_models.py

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Step 2: extract residual stream activations ---
python setup/generate_activations.py

# --- Step 3: train SAE (edit LAYER/SAE_WIDTH/SAE_TYPE in train_sae.py first) ---
python setup/train_sae.py

# --- Step 4: encode activations through the trained SAE ---
python setup/generate_sae_activations.py

# --- Step 5: run baseline classifiers (CPU only from here) ---
python probing/run_baselines.py

# --- Step 6: train probes for each setting sequentially ---
for SETTING in normal scarcity label_noise class_imbalance OOD; do
    TMPSCRIPT=$(mktemp /tmp/train_probes_XXXXXX.py)
    sed "s/^\( *\)SETTING = .*/\1SETTING = \"${SETTING}\"/" probing/train_probes.py > "$TMPSCRIPT"
    echo "Running probes: SETTING=${SETTING}"
    python "$TMPSCRIPT"
    rm -f "$TMPSCRIPT"
done

# --- Step 7: aggregate and compare results ---
python probing/combine_results.py
python probing/compare_results.py
python probing/aggregate_wins.py
