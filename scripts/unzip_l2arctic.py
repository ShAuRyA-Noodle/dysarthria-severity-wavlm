"""Parallel unzip of L2-ARCTIC speaker archives."""
from __future__ import annotations

import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "l2arctic_release_v5.0"


def _unzip(z: Path) -> tuple[str, int, str]:
    try:
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            # Skip if already extracted (top-level folder == zip stem)
            top = z.stem
            target = ROOT / top
            if target.exists() and any(target.rglob("*.wav")):
                return (z.name, len(names), "skip-already-extracted")
            zf.extractall(ROOT)
        return (z.name, len(names), "ok")
    except Exception as e:
        return (z.name, 0, f"ERR: {e}")


def main() -> None:
    zips = sorted(ROOT.glob("*.zip"))
    print(f"found {len(zips)} zips")
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_unzip, z): z for z in zips}
        for f in as_completed(futs):
            name, n, status = f.result()
            print(f"  {name:20s}  {n:6d} entries  {status}")


if __name__ == "__main__":
    main()
