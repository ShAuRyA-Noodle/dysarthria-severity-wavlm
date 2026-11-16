"""Unzip VCTK Corpus 0.92.

Extracts to DS_10283_3443/VCTK-Corpus-0.92/ next to the zip.
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "DS_10283_3443"


def main() -> None:
    zip_path = ROOT / "VCTK-Corpus-0.92.zip"
    out_dir = ROOT / "VCTK-Corpus-0.92"
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        n = len(z.namelist())
        print(f"extracting {n:,} entries to {out_dir}")
        t0 = time.time()
        z.extractall(out_dir)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
