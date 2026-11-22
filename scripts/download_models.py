"""Download pretrained encoders from Hugging Face into models/hf/<name>/ (weights + tokenizer only).

Default: the two candidate student models (small, permissive licences; see docs/DATA.md).
After this the demo and the encoding scripts work offline.
"""
import argparse

from huggingface_hub import HfApi, snapshot_download

from casetrail import config

STUDENTS = ["sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-small-v2"]
PATTERNS = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
            "model.safetensors"]
MAX_TOTAL_MB = 1000  # CLAUDE.md rule 7


def local_dir(repo_id: str):
    return config.MODEL_DIR / "hf" / repo_id.replace("/", "__")


def download_size_mb(repo_id: str) -> float:
    total = 0
    for item in HfApi().list_repo_tree(repo_id):
        size = getattr(item, "size", None)
        if size is not None and item.path in PATTERNS:
            total += size
    return total / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=STUDENTS)
    parser.add_argument("--force", action="store_true", help="allow a total download over 1 GB")
    args = parser.parse_args()
    sizes = {repo_id: download_size_mb(repo_id) for repo_id in args.models}
    for repo_id, size in sizes.items():
        print(f"{repo_id}: {size:.1f} MB")
    if sum(sizes.values()) > MAX_TOTAL_MB and not args.force:
        raise SystemExit("Total download is over 1 GB; re-run with --force after agreeing to it.")
    for repo_id in args.models:
        folder = snapshot_download(repo_id=repo_id, allow_patterns=PATTERNS, local_dir=local_dir(repo_id))
        print(f"Saved {repo_id} -> {folder}")


if __name__ == "__main__":
    main()
