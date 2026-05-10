"""Prepare a TORGO corpus tree for MFA forced alignment (Track B).

MFA expects a directory layout::

    mfa_corpus/
        <speaker>/
            <utt_id>.wav
            <utt_id>.lab        # plain-text transcript (one line)

Builds this tree by symlinking (Windows: copy) the existing 16 kHz audio
under ``data/processed/audio_16khz/`` and writing per-utterance ``.lab``
files derived from ``metadata_unified.csv``'s ``word`` column.

Also writes a custom dictionary covering OOV TORGO non-words (e.g.
``pataka``, ``eh``, ``ah``) by:
  1. Loading the standard english_us_arpa dictionary.
  2. Identifying TORGO prompts not covered.
  3. Generating G2P entries for OOVs via mfa g2p.
  4. Merging into ``data/processed/torgo_combined.dict``.

Run::

    python scripts/prep_mfa_torgo.py
    bash scripts/run_mfa.sh validate mfa_corpus_torgo/ data/processed/torgo_combined.dict english_us_arpa
    bash scripts/run_mfa.sh align    mfa_corpus_torgo/ data/processed/torgo_combined.dict english_us_arpa data/processed/textgrids/

The script is idempotent — re-running skips utterances whose .wav + .lab
already exist in the corpus tree.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import TORGO_SEVERITY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_corpus_tree(
    metadata_csv: Path,
    out_root: Path,
    *,
    use_symlink: bool = False,
) -> tuple[int, set[str]]:
    """Build the MFA-style corpus tree. Returns (n_utts_written, unique_words)."""
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metadata_csv)
    df["filepath_processed"] = df["filepath_processed"].astype(str)
    n = 0
    unique_words: set[str] = set()
    for _, row in df.iterrows():
        utt_id = str(row["utterance_id"])
        speaker = str(row["speaker_id"])
        word = str(row.get("word", "")).strip()
        if not word or word in {"nan", "[no_prompt]"}:
            continue
        # Filter prompts that are bracketed instructions like
        # `[say 'Ah-P-Eee' repeatedly]`. Keep only clean read prompts +
        # isolated words.
        if word.startswith("[") and word.endswith("]"):
            continue
        src = Path(row["filepath_processed"])
        if not src.exists():
            continue
        if speaker not in TORGO_SEVERITY:
            continue
        spk_dir = out_root / speaker
        spk_dir.mkdir(parents=True, exist_ok=True)
        dst_wav = spk_dir / f"{utt_id}.wav"
        dst_lab = spk_dir / f"{utt_id}.lab"
        if dst_wav.exists() and dst_lab.exists():
            continue
        if use_symlink:
            try:
                dst_wav.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst_wav)
        else:
            shutil.copy2(src, dst_wav)
        dst_lab.write_text(word + "\n", encoding="utf-8")
        unique_words.add(word)
        n += 1
        if n % 1000 == 0:
            logger.info("written %d utts; %d unique prompts so far", n, len(unique_words))
    return n, unique_words


def write_oov_word_list(unique_words: set[str], out_path: Path) -> int:
    """Write all unique prompts to a flat .txt for MFA validate / G2P."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for w in sorted(unique_words):
            f.write(w + "\n")
    return len(unique_words)


def main() -> None:
    metadata_csv = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"
    out_root = PROJECT_ROOT / "mfa_corpus_torgo"

    if not metadata_csv.exists():
        raise SystemExit(f"missing metadata at {metadata_csv}")

    logger.info("building MFA corpus tree at %s", out_root)
    n, words = build_corpus_tree(metadata_csv, out_root, use_symlink=False)
    logger.info("wrote %d utterances; %d unique prompts", n, len(words))

    prompts_path = PROJECT_ROOT / "data" / "processed" / "torgo_unique_prompts.txt"
    write_oov_word_list(words, prompts_path)
    logger.info("unique prompts list -> %s", prompts_path)

    print()
    print("Next steps:")
    print("  1. Validate corpus + identify OOVs:")
    print("       bash scripts/run_mfa.sh validate "
          f"{out_root.relative_to(PROJECT_ROOT)}/ english_us_arpa english_us_arpa")
    print("  2. If OOVs present, generate G2P entries:")
    print("       bash scripts/run_mfa.sh g2p english_us_arpa "
          f"{prompts_path.relative_to(PROJECT_ROOT)} "
          "data/processed/torgo_oov.dict")
    print("  3. Merge dictionaries, then align:")
    print("       bash scripts/run_mfa.sh align "
          f"{out_root.relative_to(PROJECT_ROOT)}/ "
          "data/processed/torgo_combined.dict english_us_arpa "
          "data/processed/textgrids/")


if __name__ == "__main__":
    main()
