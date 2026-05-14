import glob
import json
import os
import time

import anthropic
import pandas as pd
from tqdm import tqdm

from utils.autointerp import build_ranking_prompt, parse_ranking_response
from utils.data import MODEL_NAME

INTERP_DATA_DIR = f"data/interpretability_{MODEL_NAME}"
RESULTS_DIR     = f"results/interpretability_{MODEL_NAME}"


def _mkdirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def call_claude(
    client: anthropic.Anthropic,
    prompt: dict,
    model: str,
    max_tokens: int = 200,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    # call the Claude API with exponential-backoff retry on rate limit errors
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise
        except anthropic.APIError:
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                raise


def run_section_4_1(client: anthropic.Anthropic, model: str):
    # generate autointerp descriptions and latent rankings for the OOD pruning datasets
    print("\n" + "=" * 60)
    print("Section 4.1 — LLM: autointerp + ranking for OOD pruning")
    print("=" * 60)

    pattern = os.path.join(INTERP_DATA_DIR, "4.1_pruning", "*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  No files found at {pattern}. Run run_interpretability.py first.")
        return

    summary = {}

    for fpath in files:
        with open(fpath) as f:
            payload = json.load(f)

        dataset      = payload["metadata"]["dataset"]
        task_desc    = payload["task_description"]
        ood_desc     = payload["ood_transform"]
        latents      = payload["latents"]

        print(f"\n  Dataset: {dataset}")

        descriptions: dict[int, str] = {}
        for lat in latents:
            if lat["description"] is not None:
                descriptions[lat["latent_idx"]] = lat["description"]
                continue
            print(f"    Describing latent {lat['latent_idx']} ...")
            desc = call_claude(client, lat["autointerp_prompt"], model, max_tokens=150)
            lat["description"] = desc
            descriptions[lat["latent_idx"]] = desc
            print(f"      -> {desc[:80]}")

        ranking_prompt = build_ranking_prompt(descriptions, task_desc, ood_desc)
        print(f"    Ranking {len(descriptions)} latents …")
        ranking_text = call_claude(client, ranking_prompt, model, max_tokens=200)

        items = [(lat["latent_idx"], lat["description"]) for lat in latents]
        ranked_idxs = parse_ranking_response(ranking_text, items)
        print(f"    Ranked order: {ranked_idxs}")

        payload["ranked_latent_idxs"] = ranked_idxs
        payload["ranking_llm_response"] = ranking_text

        print(f"    OOD AUC (diff order): {payload['ood_auc_by_k_diff_order']}")

        summary[dataset] = {
            "task_description":         task_desc,
            "ood_transform":            ood_desc,
            "ranked_latent_idxs":       ranked_idxs,
            "ood_auc_by_k_diff_order":  payload["ood_auc_by_k_diff_order"],
            "latent_descriptions": {
                str(lat["latent_idx"]): lat["description"]
                for lat in latents
            },
        }

        with open(fpath, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"    Updated → {fpath}")

    out_path = os.path.join(RESULTS_DIR, "4.1_pruning_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {out_path}")

    print("\n  Dataset                        | k=1 OOD AUC | k=8 OOD AUC | Δ")
    print("  " + "-" * 70)
    for ds, res in summary.items():
        auc_k8 = res["ood_auc_by_k_diff_order"].get("8", float("nan"))
        auc_k1 = res["ood_auc_by_k_diff_order"].get("1", float("nan"))
        print(f"  {ds[:35]:35s} | {auc_k1:.4f}      | {auc_k8:.4f}      | {auc_k1 - auc_k8:+.4f}")


def run_section_4_2(client: anthropic.Anthropic, model: str):
    # generate one autointerp description per dataset for the best latent
    print("\n" + "=" * 60)
    print("Section 4.2 — LLM: autointerp descriptions per dataset")
    print("=" * 60)

    pattern = os.path.join(INTERP_DATA_DIR, "4.2_latent_investigation", "*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  No files found at {pattern}. Run run_interpretability.py first.")
        return

    rows = []

    for fpath in tqdm(files, desc="Datasets"):
        with open(fpath) as f:
            payload = json.load(f)

        dataset = payload["metadata"]["dataset"]
        best    = payload["best_latent"]

        if best["description"] is not None:
            desc = best["description"]
        else:
            desc = call_claude(client, best["autointerp_prompt"], model, max_tokens=150)
            best["description"] = desc
            with open(fpath, "w") as f:
                json.dump(payload, f, indent=2)

        rows.append({
            "dataset":     dataset,
            "latent_idx":  best["latent_idx"],
            "k1_auc":      best["k1_auc"],
            "description": desc,
            "sae_id":      payload["metadata"]["sae_id"],
            "layer":       payload["metadata"]["layer"],
        })
        tqdm.write(f"  {dataset}: latent={best['latent_idx']}  AUC={best['k1_auc']:.3f}  "
                   f'"{desc[:60]}"')

    out_path = os.path.join(RESULTS_DIR, "4.2_latent_descriptions.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  Saved {len(rows)} rows → {out_path}")


def run_section_4_3(client: anthropic.Anthropic, model: str):
    # generate autointerp descriptions for the dataset quality case studies
    print("\n" + "=" * 60)
    print("Section 4.3 — LLM: autointerp for dataset quality datasets")
    print("=" * 60)

    pattern = os.path.join(INTERP_DATA_DIR, "4.3_dataset_quality", "*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  No files found at {pattern}. Run run_interpretability.py first.")
        return

    summary = {}

    for fpath in files:
        with open(fpath) as f:
            payload = json.load(f)

        dataset = payload["metadata"]["dataset"]
        lat     = payload["top_latent"]

        print(f"\n  Dataset: {dataset}")
        print(f"    Top latent: {lat['latent_idx']}   k=1 AUC: {lat['k1_auc']:.4f}")

        if lat["description"] is None:
            desc = call_claude(client, lat["autointerp_prompt"], model, max_tokens=150)
            lat["description"] = desc
            with open(fpath, "w") as f:
                json.dump(payload, f, indent=2)
        else:
            desc = lat["description"]

        print(f"    Description: {desc}")

        disagreements = payload.get("label_disagreements", [])
        if disagreements:
            print(f"    Top label-disagreement examples (latent fires but label disagrees):")
            for d in disagreements[:5]:
                print(f"      [act={d['activation']:.2f} label={d['label']}] "
                      f"{str(d.get('text', d.get('prompt', '')))[:100]}")

        if "last_token_distribution" in payload:
            dist = payload["last_token_distribution"]
            print(f"    AI last-chars:    {dist['ai_top10'][:5]}")
            print(f"    Human last-chars: {dist['human_top10'][:5]}")

        summary[dataset] = {
            "top_latent_idx":     lat["latent_idx"],
            "k1_auc":             lat["k1_auc"],
            "description":        desc,
            "n_disagreements":    len(disagreements),
            "top_disagreements":  disagreements[:5],
        }

    out_path = os.path.join(RESULTS_DIR, "4.3_dataset_quality_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {out_path}")


if __name__ == "__main__":
    SECTION = "all"   # one of: 4.1, 4.2, 4.3, all
    MODEL = "claude-3-5-sonnet-20241022"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    _mkdirs()

    if SECTION in ("4.1", "all"):
        run_section_4_1(client, MODEL)

    if SECTION in ("4.2", "all"):
        run_section_4_2(client, MODEL)

    if SECTION in ("4.3", "all"):
        run_section_4_3(client, MODEL)

    print(f"\nLLM step done. Results in: {RESULTS_DIR}/")
