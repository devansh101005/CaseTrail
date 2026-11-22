"""Build the submission ZIP described in docs/SUBMISSION.md from the last commit.

CSD358_Devansh_Anamika_Abhinav/
  report/CaseTrail_report.pdf
  GITHUB_LINK.txt
  code/       git archive of HEAD (only committed files: no data, models, tokens or node_modules)
  dataset/    how to get IL-TUR + the exact split ids (the IL-TUR files themselves are not redistributed)
  README.txt
"""
import argparse
import io
import json
import subprocess
import zipfile

import pyarrow.parquet as pq

from casetrail import config
from casetrail.data import parquet_path

TOP = "CSD358_Devansh_Anamika_Abhinav"


def git(*args) -> str:
    result = subprocess.run(["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def repo_url() -> str:
    url = git("remote", "get-url", "origin")
    return url[:-4] if url.endswith(".git") else url


def split_ids() -> dict:
    """Document ids of every split, read from the id column only."""
    ids = {"pcr": {}, "lsi": {}}
    for split in config.PCR_SPLITS:
        for part in ["queries", "candidates"]:
            table = pq.read_table(parquet_path("pcr", f"{split}_{part}"), columns=["id"])
            ids["pcr"][f"{split}_{part}"] = table.column("id").to_pylist()
    for split in config.LSI_SPLITS:
        ids["lsi"][split] = pq.read_table(parquet_path("lsi", split), columns=["id"]).column("id").to_pylist()
    return ids


DATASET_README = """IL-TUR (Joshi et al., ACL 2024), tasks "pcr" and "lsi".

The data files are NOT included. IL-TUR is licensed CC BY-NC-SA 4.0 and is gated on Hugging Face
(you accept its conditions there), so we do not redistribute it. To get exactly the data we used:
  1. Accept the conditions at https://huggingface.co/datasets/Exploration-Lab/IL-TUR and log in
     (huggingface-cli login, or set HF_TOKEN).
  2. Run  python scripts/download_data.py   (the same script is in this folder; ~390 MB).
split_ids.json lists the id of every document in every split we used, so the splits can be checked.
"""

TOP_README = """CaseTrail: Precedent and Statute Retrieval for Indian Court Judgments
CSD358 Information Retrieval course project
Devansh Pandey (2310110461), Anamika Pal (2310110037), Abhinav Bachchas (2310110383)

report/CaseTrail_report.pdf   the project report
GITHUB_LINK.txt               link to the repository
code/                         the source code (snapshot of commit {commit})
dataset/                      how to download IL-TUR, and the ids of every split we used

To run: open code/README.md. It explains setup, how to rebuild every result from scratch, and how
to start the demo (FastAPI backend + Next.js frontend).
"""


def add_code(out: zipfile.ZipFile):
    archive = subprocess.run(["git", "archive", "--format=zip", "HEAD"], cwd=config.PROJECT_ROOT,
                             capture_output=True, check=True).stdout
    with zipfile.ZipFile(io.BytesIO(archive)) as code:
        for item in code.infolist():
            if not item.is_dir():
                out.writestr(f"{TOP}/code/{item.filename}", code.read(item))
    return len(code.infolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.PROJECT_ROOT / f"{TOP}.zip"))
    args = parser.parse_args()
    if git("status", "--porcelain"):
        raise SystemExit("Uncommitted changes: commit (and push) first, so the ZIP matches the GitHub repo.")
    report = config.PROJECT_ROOT / "docs" / "report" / "CaseTrail_report.pdf"
    if not report.exists():
        raise SystemExit(f"{report} not found.")
    commit = git("rev-parse", "--short", "HEAD")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as out:
        out.write(report, f"{TOP}/report/CaseTrail_report.pdf")
        out.writestr(f"{TOP}/GITHUB_LINK.txt", repo_url() + "\n")
        out.writestr(f"{TOP}/README.txt", TOP_README.format(commit=commit))
        out.writestr(f"{TOP}/dataset/README.txt", DATASET_README)
        out.write(config.PROJECT_ROOT / "scripts" / "download_data.py", f"{TOP}/dataset/download_data.py")
        out.writestr(f"{TOP}/dataset/split_ids.json", json.dumps(split_ids()))
        files = add_code(out)
    print(f"Wrote {args.out} (commit {commit}, {files} code files)")


if __name__ == "__main__":
    main()
