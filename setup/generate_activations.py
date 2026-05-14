import glob
import os
import random

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "gemma-3-4b-it"
HF_MODEL_ID = "google/gemma-3-4b-it"

# probe at ~21/48/74/98% depth, mirroring the original paper's layer choices
LAYER_FRACS = [0.21, 0.48, 0.74, 0.98]

torch.set_grad_enabled(False)


def select_probe_layers(num_hidden_layers: int) -> list[int]:
    # pick 4 layers spaced across model depth at the fractions defined in LAYER_FRACS
    layers = sorted({round(f * (num_hidden_layers - 1)) for f in LAYER_FRACS})
    return layers


def hook_name_for_layer(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def generate_dataset_activations(device: str = "cuda:0", max_seq_len: int = 1024, OOD: bool = False) -> None:
    # run each dataset through the model and save last-token residual stream activations per layer
    out_dir = f"data/model_activations_{MODEL_NAME}{'_OOD' if OOD else ''}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {HF_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # gemma-3 wraps its text backbone inside model.model.language_model
    if hasattr(model.model, "language_model"):
        text_model = model.model.language_model
    else:
        text_model = model.model

    num_layers = model.config.text_config.num_hidden_layers
    probe_layers = select_probe_layers(num_layers)
    print(f"  num_hidden_layers={num_layers}  ->  probe_layers={probe_layers}")

    if OOD:
        dataset_files = glob.glob("data/OOD data/*.csv")
    else:
        dataset_files = glob.glob("cleaned_data/*.csv")

    random.shuffle(dataset_files)

    for csv_path in dataset_files:
        dataset = pd.read_csv(csv_path)
        if "prompt" not in dataset.columns:
            continue

        dataset_tag = os.path.basename(csv_path).replace(".csv", "")
        if OOD:
            dataset_tag = dataset_tag.replace("_OOD", "")

        embed_fname = os.path.join(out_dir, f"{dataset_tag}_hook_embed.pt")
        layer_fnames = {
            L: os.path.join(out_dir, f"{dataset_tag}_{hook_name_for_layer(L)}.pt")
            for L in probe_layers
        }
        all_fnames = [embed_fname] + list(layer_fnames.values())

        prompts = dataset["prompt"].tolist()

        # skip if already done at the right length
        if all(os.path.exists(f) for f in all_fnames):
            try:
                existing_len = torch.load(embed_fname, weights_only=True).shape[0]
                if existing_len == len(prompts):
                    print(f"  Skipping {dataset_tag} (already done)")
                    continue
            except Exception:
                pass

        print(f"  Processing {dataset_tag} ({len(prompts)} prompts) ...")

        collected: dict[str, list[torch.Tensor]] = {
            "embed": [],
            **{str(L): [] for L in probe_layers},
        }

        def make_embed_hook():
            def _hook(module, inp, out):
                # grab the last token's embedding
                collected["embed"].append(out[:, -1, :].float().cpu())
            return _hook

        def make_layer_hook(layer_idx: int):
            def _hook(module, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                collected[str(layer_idx)].append(hidden[:, -1, :].float().cpu())
            return _hook

        hooks = []
        hooks.append(text_model.embed_tokens.register_forward_hook(make_embed_hook()))
        for L in probe_layers:
            hooks.append(text_model.layers[L].register_forward_hook(make_layer_hook(L)))

        for i in tqdm(range(0, len(prompts), 1), desc=dataset_tag, leave=False):
            batch_prompts = prompts[i : i + 1]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            input_ids = encoded["input_ids"].to(device)
            with torch.no_grad():
                model(input_ids=input_ids)

        for h in hooks:
            h.remove()

        embed_tensor = torch.cat(collected["embed"], dim=0)
        torch.save(embed_tensor, embed_fname)

        for L in probe_layers:
            layer_tensor = torch.cat(collected[str(L)], dim=0)
            torch.save(layer_tensor, layer_fnames[L])

        print(f"  Saved {dataset_tag}: shape {embed_tensor.shape}")


if __name__ == "__main__":
    DEVICE = "cuda:0"
    MAX_SEQ_LEN = 1024
    OOD = False  # set to True to generate OOD activations instead

    generate_dataset_activations(device=DEVICE, max_seq_len=MAX_SEQ_LEN, OOD=OOD)
