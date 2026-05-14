#!/bin/bash -l
#SBATCH -p ecsstudents_l4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=06:00:00
#SBATCH -o logs/pipeline_stitching_%j.out
#SBATCH -e logs/pipeline_stitching_%j.err
#SBATCH -J pipeline_stitching

# Step 10: cross-modal stitching extension
# Can be submitted alongside pipeline_setup.sh after step 4 completes,
# or run independently once SAE activations exist.

module load conda/python3
conda init
source ~/.bashrc
conda activate SAE

export HF_TOKEN=hf_PUT_TOKEN_HERE
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs
cd SAE
python extensions/model_stitching.py
