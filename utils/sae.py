import glob
import json
import os

import torch
from tqdm import tqdm

MODEL_NAME = "gemma-3-4b-it"
DEFAULT_SAE_WIDTH = 16384   # 16k features
DEFAULT_SAE_K = 32          # top-k sparsity


def sae_id_to_path(sae_id: str, model_name: str = MODEL_NAME) -> str:
    return f"data/saes_{model_name}/{sae_id}/ae.pt"


def get_gemma3_sae_ids(layer: int) -> list[str]:
    # scan disk for all trained SAE checkpoints at this layer
    pattern = f"data/saes_{MODEL_NAME}/layer_{layer}/width_*/*/ae.pt"
    ids = []
    for p in sorted(glob.glob(pattern)):
        parts = p.replace("\\", "/").split("/")
        ids.append("/".join(parts[-4:-1]))   # e.g. "layer_22/width_16k/topk_32"
    return ids


def layer_to_sae_ids_gemma3(layer: int) -> list[str]:
    ids = get_gemma3_sae_ids(layer)
    if not ids:
        return [f"layer_{layer}/width_16k/topk_{DEFAULT_SAE_K}"]
    return ids


def load_gemma3_sae(sae_id: str):
    # load an SAE checkpoint, picking the right class based on the type string in sae_id
    path = sae_id_to_path(sae_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SAE not found at {path}.\n"
            "Run  python setup/train_sae.py  first."
        )

    sae_dir  = os.path.dirname(path)
    type_str = sae_id.split("/")[-1]

    config: dict = {}
    cfg_path = os.path.join(sae_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)

    try:
        if type_str.startswith("topk") or type_str.startswith("batch_topk"):
            k = config.get("k", int(type_str.split("_")[-1]))
            from dictionary_learning.trainers.top_k import AutoEncoderTopK
            sae = AutoEncoderTopK.from_pretrained(path, k=k)

        elif type_str.startswith("jumprelu"):
            from dictionary_learning.dictionary import JumpReluAutoEncoder
            sae = JumpReluAutoEncoder.from_pretrained(path)

        elif type_str in ("standard", "standard_new", "p_anneal"):
            from dictionary_learning.dictionary import AutoEncoder
            sae = AutoEncoder.from_pretrained(path)

        elif type_str == "april_update":
            try:
                from dictionary_learning.dictionary import AutoEncoderNew
                sae = AutoEncoderNew.from_pretrained(path)
            except (ImportError, AttributeError):
                from dictionary_learning.dictionary import AutoEncoder
                sae = AutoEncoder.from_pretrained(path)

        elif type_str == "gated":
            from dictionary_learning.dictionary import GatedAutoEncoder
            sae = GatedAutoEncoder.from_pretrained(path)

        elif type_str.startswith("matryoshka_batch_topk"):
            k = config.get("k", int(type_str.split("_")[-1]))
            from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
            sae = MatryoshkaBatchTopKSAE.from_pretrained(path, k=k)

        else:
            raise ValueError(
                f"Unrecognised SAE type '{type_str}' in sae_id '{sae_id}'.\n"
                "Expected one of: topk_<k>, batch_topk_<k>, jumprelu_<k>, "
                "standard, standard_new, april_update, gated, p_anneal."
            )
    except ImportError as exc:
        raise ImportError(
            f"dictionary_learning not installed or component missing: {exc}\n"
            "Install: pip install git+https://github.com/saprmarks/dictionary_learning"
        ) from exc

    sae.eval()
    return sae


def sae_id_to_sae_gemma3(sae_id: str, device: str = "cpu"):
    return load_gemma3_sae(sae_id).to(device)


def get_sae_layers_gemma3(all_layers: bool = False) -> list[int]:
    # return all integer probe layers, or just the default SAE layer
    from utils.data import get_layers, get_default_sae_layer
    if all_layers:
        return [l for l in get_layers() if l != "embed"]
    return [get_default_sae_layer()]


def collect_all_activations(layer, model_name: str = MODEL_NAME) -> torch.Tensor:
    # concatenate all saved residual-stream activations for a given layer
    if layer == "embed":
        pattern = f"data/model_activations_{model_name}/*_hook_embed.pt"
    else:
        pattern = f"data/model_activations_{model_name}/*_blocks.{layer}.hook_resid_post.pt"

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No activation files found for layer {layer} in {model_name}"
        )

    tensors = []
    for p in tqdm(paths, desc=f"Loading layer-{layer} activations"):
        tensors.append(torch.load(p, weights_only=True).float())
    return torch.cat(tensors, dim=0)
