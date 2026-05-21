"""One-shot dataset download for TEMImageNet v1.3 (Lin et al., Sci. Rep. 2021).

Clones https://github.com/xinhuolin/TEM-ImageNet-v1.3 (shallow, ~2 GB) into
data/raw/temimagenet/ on first run. Idempotent: re-runs after a successful clone
exit early.

Run with:
    uv run python data/download.py
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/xinhuolin/TEM-ImageNet-v1.3.git"
DEFAULT_DEST = Path("data/raw/temimagenet")
EXPECTED_SUBDIRS = ("image", "circularMask")
PRESENT_THRESHOLD = 1000  # conservative; full library is ~10k+

logger = logging.getLogger(__name__)


def _already_present(dest: Path) -> bool:
    for sub in EXPECTED_SUBDIRS:
        d = dest / sub
        if not d.is_dir():
            return False
        if sum(1 for _ in d.glob("*.png")) < PRESENT_THRESHOLD:
            return False
    return True


def main(dest: Path = DEFAULT_DEST) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if shutil.which("git") is None:
        raise RuntimeError("git not found on PATH; install git to fetch the dataset")

    if _already_present(dest):
        logger.info("dataset already present at %s; skipping clone", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise RuntimeError(
            f"{dest} exists but does not look like a complete TEMImageNet clone; "
            f"remove it and re-run"
        )

    logger.info("cloning %s into %s (shallow, ~2 GB)", REPO_URL, dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(dest)],
        check=True,
    )
    logger.info("done")


if __name__ == "__main__":
    main()
    sys.exit(0)
