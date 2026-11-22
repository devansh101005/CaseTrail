"""Download the IL-TUR `pcr` and `lsi` tasks from Hugging Face into data/raw/ (about 400 MB).

IL-TUR is gated: accept the conditions on https://huggingface.co/datasets/Exploration-Lab/IL-TUR
and log in once with `hf auth login` (or set the HF_TOKEN environment variable). No token is stored here.
We download the plain parquet files directly instead of the card's `revision="script"` loader,
because recent `datasets` versions no longer run dataset scripts (see docs/DECISIONS.md).
"""
import argparse

from huggingface_hub import HfApi, snapshot_download

from casetrail import config

MAX_TOTAL_MB = 1000  # stop before downloading more than ~1 GB


def remote_size_mb(tasks: list) -> float:
    api = HfApi()
    total = 0
    for item in api.list_repo_tree(config.HF_DATASET, repo_type="dataset", recursive=True):
        size = getattr(item, "size", None)
        if size is not None and item.path.split("/")[0] in tasks:
            total += size
    return total / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=config.TASKS, help="IL-TUR task folders to download")
    parser.add_argument("--force", action="store_true", help="allow a download larger than 1 GB")
    args = parser.parse_args()

    size_mb = remote_size_mb(args.tasks)
    print(f"Tasks {args.tasks}: {size_mb:.1f} MB on the Hub")
    if size_mb > MAX_TOTAL_MB and not args.force:
        raise SystemExit("Download is over 1 GB; re-run with --force after agreeing to it.")

    patterns = [f"{task}/*" for task in args.tasks]
    folder = snapshot_download(
        repo_id=config.HF_DATASET,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=config.RAW_DIR,
    )
    print(f"Saved to {folder}")
    for path in sorted(config.RAW_DIR.glob("*/*.parquet")):
        print(f"  {path.relative_to(config.RAW_DIR)}  {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
