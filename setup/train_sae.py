import os
import shutil
import sys
from dataclasses import dataclass, asdict, field
from typing import Optional, Type, Any

import torch

from utils.data import get_layers, MODEL_NAME
from utils.sae import (
    collect_all_activations,
    sae_id_to_path,
    DEFAULT_SAE_K,
    DEFAULT_SAE_WIDTH,
)

try:
    from dictionary_learning.training import trainSAE
except ImportError:
    print("ERROR: dictionary_learning is not installed.")
    print("  pip install git+https://github.com/saprmarks/dictionary_learning")
    sys.exit(1)

from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer, BatchTopKSAE
from dictionary_learning.trainers.standard import StandardTrainer, StandardTrainerAprilUpdate
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKTrainer,
    MatryoshkaBatchTopKSAE,
)
from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    JumpReluAutoEncoder,
)


# dataclasses for each trainer config, matching demo_config.py exactly

@dataclass
class BaseTrainerConfig:
    activation_dim: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: Optional[str]
    warmup_steps: int
    steps: int
    decay_start: Optional[int]


@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    resample_steps: Optional[int] = None


@dataclass
class StandardNewTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class PAnnealTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    initial_sparsity_penalty: float
    sparsity_warmup_steps: Optional[int]
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: Optional[int] = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    k_anneal_steps: Optional[int] = None


@dataclass
class MatryoshkaBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    group_fractions: list[float] = field(
        default_factory=lambda: [
            (1 / 32), (1 / 16), (1 / 8), (1 / 4), ((1 / 2) + (1 / 32)),
        ]
    )
    group_weights: Optional[list[float]] = None
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    k_anneal_steps: Optional[int] = None


@dataclass
class GatedTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    target_l0: int
    sparsity_warmup_steps: Optional[int]
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001


# training hyperparameters (matching demo_config.py)
WARMUP_STEPS = 1000
SPARSITY_WARMUP_STEPS = 5000
DECAY_START_FRACTION = 0.8
K_ANNEAL_END_FRACTION = 0.01
LEARNING_RATE = 5e-5
SAE_BATCH_SIZE = 2048
SEED = 0

DEFAULT_L1_STANDARD = 0.02
DEFAULT_L1_P_ANNEAL = 0.01
DEFAULT_L1_GATED    = 0.04

_K_BASED_TYPES = {"topk", "batch_topk", "jumprelu", "matryoshka_batch_topk"}

TRAINER_REGISTRY: dict[str, tuple] = {
    "topk":                  (TopKTrainer,                AutoEncoderTopK),
    "batch_topk":            (BatchTopKTrainer,           BatchTopKSAE),
    "standard":              (StandardTrainer,            AutoEncoder),
    "standard_new":          (StandardTrainerAprilUpdate, AutoEncoder),
    "gated":                 (GatedSAETrainer,            GatedAutoEncoder),
    "p_anneal":              (PAnnealTrainer,             AutoEncoder),
    "jumprelu":              (JumpReluTrainer,            JumpReluAutoEncoder),
    "matryoshka_batch_topk": (MatryoshkaBatchTopKTrainer, MatryoshkaBatchTopKSAE),
}

ALL_TYPES: list[str] = list(TRAINER_REGISTRY.keys())
print(f"Available SAE types: {ALL_TYPES}")


def type_to_id_suffix(sae_type: str, k: int) -> str:
    return f"{sae_type}_{k}" if sae_type in _K_BASED_TYPES else sae_type


def cyclic_iter(activations: torch.Tensor, batch_size: int, device: str):
    # infinite cyclic batch iterator over in-memory activations
    N = activations.shape[0]
    while True:
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            yield activations[perm[i: i + batch_size]].to(device)


def _build_config(sae_type: str, d_model: int, layer: int, width: int, k: int, device: str, steps: int) -> dict:
    # construct the trainer config dict expected by dictionary_learning's trainSAE
    trainer_cls, dict_cls = TRAINER_REGISTRY[sae_type]
    decay_start = min(int(steps * DECAY_START_FRACTION), steps - 1)
    sparsity_warmup = min(SPARSITY_WARMUP_STEPS, decay_start - 1)
    k_anneal_end = int(steps * K_ANNEAL_END_FRACTION)
    layer_str = str(layer)
    submodule_name = f"layer_{layer}"

    if sae_type in ("topk", "batch_topk"):
        cfg = TopKTrainerConfig(
            activation_dim=d_model, device=device, layer=layer_str, lm_name=MODEL_NAME,
            submodule_name=submodule_name, trainer=trainer_cls, dict_class=dict_cls,
            wandb_name=None, warmup_steps=WARMUP_STEPS, steps=steps, decay_start=decay_start,
            dict_size=width, seed=SEED, lr=LEARNING_RATE, k=k, k_anneal_steps=k_anneal_end,
        )
    elif sae_type == "matryoshka_batch_topk":
        cfg = MatryoshkaBatchTopKTrainerConfig(
            activation_dim=d_model, device=device, layer=layer_str, lm_name=MODEL_NAME,
            submodule_name=submodule_name, trainer=trainer_cls, dict_class=dict_cls,
            wandb_name=None, warmup_steps=WARMUP_STEPS, steps=steps, decay_start=decay_start,
            dict_size=width, seed=SEED, lr=LEARNING_RATE, k=k, k_anneal_steps=k_anneal_end,
        )
    elif sae_type == "standard":
        cfg = StandardTrainerConfig(
            activation_dim=d_model, device=device, layer=layer_str, lm_name=MODEL_NAME,
            submodule_name=submodule_name, trainer=trainer_cls, dict_class=dict_cls,
            wandb_name=None, warmup_steps=WARMUP_STEPS, steps=steps, decay_start=decay_start,
            dict_size=width, seed=SEED, lr=LEARNING_RATE,
            l1_penalty=DEFAULT_L1_STANDARD, sparsity_warmup_steps=sparsity_warmup,
        )
    elif sae_type == "standard_new":
        cfg = StandardNewTrainerConfig(
            activation_dim=d_model, device=device, layer=layer_str, lm_name=MODEL_NAME,
            submodule_name=submodule_name, trainer=trainer_cls, dict_class=dict_cls,
            wandb_name=None, warmup_steps=WARMUP_STEPS, steps=steps, decay_start=decay_start,
            dict_size=width, seed=SEED, lr=LEARNING_RATE,
            l1_penalty=DEFAULT_L1_STANDARD, sparsity_warmup_steps=sparsity_warmup,
        )
    elif sae_type == "gated":
        cfg = GatedTrainerConfig(
            activation_dim=d_model, device=device, layer=layer_str, lm_name=MODEL_NAME,
            submodule_name=submodule_name, trainer=trainer_cls, dict_class=dict_cls,
            wandb_name=None, warmup_steps=WARMUP_STEPS, steps=steps, decay_start=decay_start,
            dict_size=width, seed=SEED, lr=LEARNING_RATE,
            l1_penalty=DEFAULT_L1_GATED, sparsity_warmup_steps=sparsity_warmup,
        )
    elif sae_type == "p_anneal":
        cfg = PAnnealTrainerConfig(
            activation_dim=d_model, device=device, layer=layer_str, lm_name=MODEL_NAME,
            submodule_name=submodule_name, trainer=trainer_cls, dict_class=dict_cls,
            wandb_name=None, warmup_steps=WARMUP_STEPS, steps=steps, decay_start=decay_start,
            dict_size=width, seed=SEED, lr=LEARNING_RATE,
            initial_sparsity_penalty=DEFAULT_L1_P_ANNEAL, sparsity_warmup_steps=sparsity_warmup,
        )
    elif sae_type == "jumprelu":
        cfg = JumpReluTrainerConfig(
            activation_dim=d_model, device=device, layer=layer_str, lm_name=MODEL_NAME,
            submodule_name=submodule_name, trainer=trainer_cls, dict_class=dict_cls,
            wandb_name=None, warmup_steps=WARMUP_STEPS, steps=steps, decay_start=decay_start,
            dict_size=width, seed=SEED, lr=LEARNING_RATE,
            target_l0=k, sparsity_warmup_steps=sparsity_warmup,
        )
    else:
        raise ValueError(f"Unknown sae_type: {sae_type}")

    return asdict(cfg)


def _find_ae_pt(tmp_dir: str) -> str:
    # trainSAE may put ae.pt in trainer_0/ or directly in the save dir
    for candidate in (
        os.path.join(tmp_dir, "trainer_0", "ae.pt"),
        os.path.join(tmp_dir, "ae.pt"),
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"ae.pt not found anywhere in {tmp_dir} after training")


def _train_sae(activations: torch.Tensor, sae_type: str, layer: int, device: str, k: int, width: int, steps: int) -> None:
    # train one SAE type and move the checkpoint to its canonical save path
    suffix    = type_to_id_suffix(sae_type, k)
    sae_id    = f"layer_{layer}/width_16k/{suffix}"
    save_path = sae_id_to_path(sae_id)

    if os.path.exists(save_path):
        print(f"    Already exists: {save_path}  (skipping)")
        return

    print(f"    Training {sae_type} ...")
    d_model   = activations.shape[1]
    data_iter = cyclic_iter(activations, batch_size=SAE_BATCH_SIZE, device=device)
    config    = _build_config(sae_type, d_model, layer, width, k, device, steps)

    sae_dir = os.path.dirname(save_path)
    tmp_dir = sae_dir + "_training_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        trainSAE(
            data=data_iter,
            trainer_configs=[config],
            steps=steps,
            save_dir=tmp_dir,
            log_steps=max(steps // 20, 200),
            verbose=True,
            device=device,
        )

        ae_src = _find_ae_pt(tmp_dir)
        os.makedirs(sae_dir, exist_ok=True)
        shutil.move(ae_src, save_path)

        cfg_src = os.path.join(os.path.dirname(ae_src), "config.json")
        if os.path.exists(cfg_src):
            shutil.move(cfg_src, os.path.join(sae_dir, "config.json"))

        print(f"    Saved -> {save_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def train_layer(layer: int, sae_types: list[str], device: str, k: int, width: int, steps: int) -> None:
    # load activations once per layer, then train all requested SAE types on them
    print(f"\nLayer {layer}")

    to_train = [
        t for t in sae_types
        if t in TRAINER_REGISTRY
        and not os.path.exists(sae_id_to_path(f"layer_{layer}/width_16k/{type_to_id_suffix(t, k)}"))
    ]

    if not to_train:
        print(f"  All SAE types already trained for layer {layer} - skipping")
        return

    print(f"  Loading activations (layer {layer}) ...")
    activations = collect_all_activations(layer)
    print(f"  Shape: {activations.shape}  (N x d_model)")

    for sae_type in to_train:
        _train_sae(activations, sae_type, layer, device, k, width, steps)


def train_all(sae_types: list[str], device: str, k: int, width: int, steps: int) -> None:
    # train all SAE types across all integer probe layers
    layers: list[int] = [l for l in get_layers() if isinstance(l, int)]
    print(f"Training SAE types {sae_types} for layers: {layers}")
    for layer in layers:
        train_layer(layer, sae_types, device, k, width, steps)
    print("\nAll SAEs trained.")


if __name__ == "__main__":
    DEVICE = "cuda:0"
    LAYER = None          # set to an int to train a single layer, or None for all layers
    SAE_TYPE = None       # set to a specific type (e.g. "topk"), or None for all types
    K = DEFAULT_SAE_K     # top-k sparsity for k-based SAEs
    WIDTH = DEFAULT_SAE_WIDTH
    STEPS = 30000

    sae_types = [SAE_TYPE] if SAE_TYPE else ALL_TYPES

    if LAYER is not None:
        train_layer(LAYER, sae_types, DEVICE, K, WIDTH, STEPS)
    else:
        train_all(sae_types, DEVICE, K, WIDTH, STEPS)
