#!/bin/bash -l
#SBATCH -p ecsstudents_l4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --time=08:00:00
#SBATCH -o logs/pipeline_interp_%j.out
#SBATCH -e logs/pipeline_interp_%j.err
#SBATCH -J pipeline_interp

# Steps 8-9: interpretability analyses + LLM autointerp
#
# run_llm_interp.py calls the Anthropic API — requires outbound internet.
# Set ANTHROPIC_API_KEY below before submitting.

module load conda/python3
conda init
source ~/.bashrc
conda activate SAE

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ANTHROPIC_API_KEY=sk-ant-PUT_KEY_HERE

mkdir -p logs
cd SAE

# --- Step 8: non-LLM interpretability ---
python interpretability/run_interpretability.py
python interpretability/js_separability.py
python interpretability/cross_concept.py
python interpretability/feature_comparison.py
python interpretability/plot_comparison.py

# --- Step 9: LLM-based latent descriptions (needs internet for Anthropic API) ---
python interpretability/run_llm_interp.py
