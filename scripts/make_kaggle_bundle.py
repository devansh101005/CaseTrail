"""Zip the code needed by the Kaggle notebook -> data/kaggle_bundle/casetrail_code.zip (upload it as a Kaggle dataset).

Contains src/, scripts/, results/tuned_params.json, pyproject.toml, requirements.txt. No data, models, indexes
or tokens: the notebook downloads IL-TUR itself with the HF_TOKEN Kaggle secret.
"""
import argparse
import zipfile

from casetrail import config

INCLUDE_DIRS = ["src", "scripts"]
INCLUDE_FILES = ["pyproject.toml", "requirements.txt", "results/tuned_params.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    params = config.RESULTS_DIR / "tuned_params.json"
    if not params.exists():
        raise SystemExit("results/tuned_params.json missing: run scripts/tune_pcr_lexical.py first.")
    out_dir = config.DATA_DIR / "kaggle_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "casetrail_code.zip"
    count = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for folder in INCLUDE_DIRS:
            for path in sorted((config.PROJECT_ROOT / folder).rglob("*.py")):
                bundle.write(path, path.relative_to(config.PROJECT_ROOT).as_posix())
                count += 1
        for name in INCLUDE_FILES:
            bundle.write(config.PROJECT_ROOT / name, name)
            count += 1
    print(f"Wrote {out_path} ({count} files, {out_path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
