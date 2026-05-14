# Sparse Encoder Probing

Probing experiments on Sparse Autoencoder (SAE) latents extracted from `gemma-3-4b-it` residual stream activations. The pipeline covers binary classification probing across 160+ datasets under four experimental settings (normal, data scarcity, class imbalance, label noise), plus interpretability analyses and a cross-modal stitching extension.

---

## Project Structure

```
.
├── utils/                        # Shared helpers
│   ├── data.py                   # Dataset loading, activation paths, train/test splits
│   ├── sae.py                    # SAE loading and layer/ID helpers
│   ├── training.py               # CV-tuned probe trainers (logistic, ridge, KNN, XGBoost, MLP)
│   └── autointerp.py             # SAE encoding, latent ranking, Claude autointerp prompts
├── setup/                        # One-time setup scripts
│   ├── download_models.py        # Download gemma-3-4b-it weights from HuggingFace
│   ├── generate_activations.py   # Extract and save residual stream activations
│   ├── generate_sae_activations.py  # Encode activations through a trained SAE
│   └── train_sae.py              # Train SAEs (topk, gated, p_anneal, matryoshka, …)
├── probing/                      # Probing experiments
│   ├── run_baselines.py          # Logistic regression / KNN / MLP baselines
│   ├── train_probes.py           # Sparse linear probes over top-k SAE latents
│   ├── compare_results.py        # SAE probe vs baseline comparison tables
│   ├── combine_results.py        # Merge per-setting result files
│   └── aggregate_wins.py         # Win-rate statistics across settings
├── interpretability/             # Interpretability analyses
│   ├── run_interpretability.py   # Generate prompts for autointerp
│   ├── run_llm_interp.py         # Claude-based latent description and ranking
│   ├── js_separability.py        # JS separability, salient neuron overlap
│   ├── cross_concept.py          # Cross-concept cluster separation (CH, DB, silhouette)
│   ├── feature_comparison.py     # Gini selectivity and Jaccard feature-reuse analysis
│   └── plot_comparison.py        # Plots for selectivity and feature-reuse results
├── extensions/                   # Extension experiments
│   ├── model_stitching.py        # Cross-modal stitching: SAE text -> CIFAR-100 image space
│   └── run_pipeline.py           # End-to-end pipeline: generate activations + train probes
└── requirements.txt
```

---

## Installation

**Python 3.10+ required.**

```bash
git clone <repo-url>
cd sparse-encoder
pip install -r requirements.txt
```

The `dictionary_learning` package is Anthropic's open-source SAE library:

```bash
pip install git+https://github.com/EleutherAI/dictionary_learning.git
```

Gemma-3 is a gated model. Accept the licence on [huggingface.co](https://huggingface.co/google/gemma-3-4b-it) then authenticate:

```bash
huggingface-cli login
```

---

## Running the Pipeline

### 1. Download model weights

```bash
python setup/download_models.py
```

### 2. Generate residual stream activations

Saves per-layer `.pt` files under `data/model_activations_gemma-3-4b-it/`.

```bash
python setup/generate_activations.py
```

### 3. Train SAEs

Edit the constants at the top of `__main__` in `setup/train_sae.py` (layer, width, SAE type), then:

```bash
python setup/train_sae.py
```

Checkpoints are saved to `data/saes_gemma-3-4b-it/layer_<N>/width_<W>/<type>/ae.pt`.

### 4. Run probing experiments

**All settings end-to-end (generates SAE activations on-the-fly, trains probes and baselines):**

```bash
python extensions/run_pipeline.py
```

**Or run each stage separately:**

```bash
python setup/generate_sae_activations.py   # encode activations through SAE
python probing/run_baselines.py            # train baseline classifiers
python probing/train_probes.py             # train sparse SAE probes
```

**Aggregate and compare:**

```bash
python probing/combine_results.py          # merge result files
python probing/compare_results.py          # build comparison tables
python probing/aggregate_wins.py           # compute win-rate statistics
```

### 5. Interpretability analyses

```bash
python interpretability/run_interpretability.py   # JS separability, autointerp data prep
python interpretability/run_llm_interp.py         # Claude latent descriptions (needs ANTHROPIC_API_KEY)
python interpretability/js_separability.py        # JS separability vs logistic regression
python interpretability/cross_concept.py          # cross-concept cluster metrics
python interpretability/feature_comparison.py     # Gini selectivity + Jaccard feature reuse
python interpretability/plot_comparison.py        # generate comparison figures
```

`run_llm_interp.py` requires:

```bash
export ANTHROPIC_API_KEY=sk-...
```

### 6. Cross-modal stitching extension

Fits a ridge regression map from SAE text latents to ResNet50 CIFAR-100 image features and evaluates 1-NN accuracy.

```bash
python extensions/model_stitching.py
```

---

## Data Layout

Expected directory structure for pre-saved data (paths set in `utils/data.py`):

```
data/
├── probing_datasets_MASTER.csv          # master list of probing tasks
├── model_activations_gemma-3-4b-it/     # residual stream activations (.pt per dataset/layer)
├── saes_gemma-3-4b-it/                  # SAE checkpoints
│   └── layer_<N>/width_<W>/<type>/ae.pt
└── sae_activations_gemma-3-4b-it/       # encoded SAE activations (generated on-the-fly)

cleaned_data/                            # per-dataset CSVs with text and target columns
results/                                 # output tables and figures (generated at runtime)
```

---

## Configuration

Each script reads its settings from hardcoded variables at the top of its `if __name__ == "__main__":` block, Edit those variables directly before running.

Key settings:

| Variable | File | Default | Description |
|---|---|---|---|
| `DEVICE` | most scripts | `"cuda"` | PyTorch device |
| `LAYER` | most scripts | `None` (auto) | SAE layer index |
| `SETTING` | `probing/train_probes.py` | `"normal"` | `normal`, `scarcity`, `imbalance`, `label_noise` |
| `MODEL` | `interpretability/run_llm_interp.py` | `claude-3-5-sonnet-20241022` | Claude model for autointerp |
| `MAX_CONCEPTS` | `extensions/model_stitching.py` | `100` | Number of CIFAR-100 classes to stitch |
| `SAE_ACT_BASEPATH` | `utils/data.py` | `"scratch/data"` | **Temporary** path for SAE activations, change this to a persistent location before long runs |
