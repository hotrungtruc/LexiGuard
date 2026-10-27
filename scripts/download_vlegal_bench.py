"""
Download Script: CMC-OPENAI/VLegal-Bench Dataset
=================================================
Downloads the complete VLegal-Bench dataset from Hugging Face to data/vlegal_bench.
"""

import sys
import time
from pathlib import Path
from loguru import logger
from huggingface_hub import snapshot_download

# Windows console encoding fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TARGET_DIR = Path("data/vlegal_bench")
REPO_ID = "CMC-OPENAI/VLegal-Bench"


def main():
    logger.info(f"Starting download of dataset '{REPO_ID}' to '{TARGET_DIR.resolve()}'...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    try:
        downloaded_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(TARGET_DIR),
            local_dir_use_symlinks=False,
            max_workers=4,
        )
        elapsed = time.time() - start_time
        logger.info(f"Dataset successfully downloaded to: {downloaded_path} in {elapsed:.2f}s.")

        # Summary of downloaded files
        all_files = list(TARGET_DIR.rglob("*"))
        jsonl_files = list(TARGET_DIR.rglob("*.jsonl"))
        py_files = list(TARGET_DIR.rglob("*.py"))
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())

        logger.info(f"Total files: {len([f for f in all_files if f.is_file()])}")
        logger.info(f"Total JSONL tasks: {len(jsonl_files)}")
        logger.info(f"Total size: {total_size / (1024 * 1024):.2f} MB")

        print("\n--- VLegal-Bench Task Breakdown ---")
        for jf in sorted(jsonl_files):
            # Count records
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
                print(f"- {jf.relative_to(TARGET_DIR)}: {count} samples")
            except Exception as e:
                print(f"- {jf.relative_to(TARGET_DIR)}: Error reading ({e})")

    except Exception as exc:
        logger.error(f"Failed to download dataset: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
