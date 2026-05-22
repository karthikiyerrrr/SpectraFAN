"""Generate the committed TEMImageNet v1 split.

One-shot script. Reads ``data/raw/temimagenet`` via list_pairs, shuffles stems
with seed=0, writes ``data/splits/temimagenet_v1/{train,val,test}.txt``.

Idempotency: if the split files already exist and have identical contents to
the freshly computed split, no-op. If they exist with different contents, raise
loudly -- accidental split rotation invalidates every prior run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from spectrafan.data import build_split, list_pairs

SPLITS_DIR = Path("data/splits/temimagenet_v1")
DATA_ROOT = Path("data/raw/temimagenet")
SEED = 0


def main() -> None:
    if not DATA_ROOT.is_dir():
        sys.exit(f"missing dataset root: {DATA_ROOT.resolve()} -- run data/download.py first")

    df = list_pairs(DATA_ROOT)
    stems = df["stem"].to_list()
    train, val, test = build_split(stems, seed=SEED)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in (("train", train), ("val", val), ("test", test)):
        path = SPLITS_DIR / f"{name}.txt"
        new_content = "\n".join(items) + "\n"
        if path.exists():
            existing = path.read_text()
            if existing == new_content:
                print(f"unchanged: {path}")
                continue
            sys.exit(
                f"refusing to overwrite {path} with different contents -- "
                "delete it explicitly if you really want to rotate the split"
            )
        path.write_text(new_content)
        print(f"wrote {len(items):>5d} stems to {path}")


if __name__ == "__main__":
    main()
