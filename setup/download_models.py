import sys

# models needed by the pipeline
MODELS = [
    "google/gemma-3-4b-it",
]


def download_all(cache_dir=None):
    # download all required HuggingFace model weights via snapshot_download
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.  Run:  pip install huggingface_hub")
        sys.exit(1)

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    for repo_id in MODELS:
        print(f"\nDownloading {repo_id} ...")
        try:
            local_path = snapshot_download(repo_id=repo_id, **kwargs)
            print(f"   Cached at: {local_path}")
        except Exception as exc:
            print(f"   FAILED: {exc}")
            print(
                "\n   If this is a gated model, accept the licence on https://huggingface.co "
                "and authenticate:\n"
                "       huggingface-cli login\n"
                "   or set HF_TOKEN in your environment."
            )
            sys.exit(1)

    print("\n=== All downloads complete. ===")
    print(
        "On an air-gapped machine, set:\n"
        "    export HF_DATASETS_OFFLINE=1\n"
        "    export TRANSFORMERS_OFFLINE=1"
    )


if __name__ == "__main__":
    # set cache_dir to a custom path if you want to store models somewhere specific
    CACHE_DIR = None

    download_all(CACHE_DIR)
