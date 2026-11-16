"""L2-ARCTIC parser.

Produces metadata + per-speaker phone-error-rate (PER) from manual annotations.
PER serves as accent severity proxy for ATD specificity test against
dysarthric severity.

Output:
  data/processed/metadata/metadata_l2arctic.csv
    columns: speaker_id, l1, gender, sentence_id, wav_path,
             textgrid_path, annotation_path, has_annotation, duration_s

  results/l2arctic_speaker_per.csv
    columns: speaker_id, l1, gender, n_annotated, n_subs, n_dels, n_adds,
             n_phones_total, per (errors / total phones)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from praatio import textgrid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

L2_ROOT = PROJECT_ROOT / "l2arctic_release_v5.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SPEAKER_INFO = {
    "ABA":   ("M", "Arabic"),     "SKA":   ("F", "Arabic"),
    "YBAA":  ("M", "Arabic"),     "ZHAA":  ("F", "Arabic"),
    "BWC":   ("M", "Chinese"),    "LXC":   ("F", "Chinese"),
    "NCC":   ("F", "Chinese"),    "TXHC":  ("M", "Chinese"),
    "ASI":   ("M", "Hindi"),      "RRBI":  ("M", "Hindi"),
    "SVBI":  ("F", "Hindi"),      "TNI":   ("F", "Hindi"),
    "HJK":   ("F", "Korean"),     "HKK":   ("M", "Korean"),
    "YDCK":  ("F", "Korean"),     "YKWK":  ("M", "Korean"),
    "EBVS":  ("M", "Spanish"),    "ERMS":  ("M", "Spanish"),
    "MBMPS": ("F", "Spanish"),    "NJS":   ("F", "Spanish"),
    "HQTV":  ("M", "Vietnamese"), "PNV":   ("F", "Vietnamese"),
    "THV":   ("F", "Vietnamese"), "TLV":   ("M", "Vietnamese"),
}


def _parse_annotation(tg_path: Path) -> tuple[int, int, int, int]:
    """Return (n_subs, n_dels, n_adds, n_phones_total) from manual annot."""
    try:
        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    except Exception as exc:
        logger.warning("annotation parse failed: %s: %s", tg_path.name, exc)
        return 0, 0, 0, 0
    if "phones" not in tg.tierNames:
        return 0, 0, 0, 0
    n_sub = n_del = n_add = n_total = 0
    for interval in tg.getTier("phones").entries:
        label = (interval.label or "").strip()
        if not label or label in ("sil", "sp", ""):
            continue
        n_total += 1
        # Annotation format: CPL,PPL,s | sil,PPL,a | CPL,sil,d
        if "," in label:
            parts = label.split(",")
            if len(parts) == 3:
                op = parts[2].strip().lower()
                if op == "s":
                    n_sub += 1
                elif op == "d":
                    n_del += 1
                elif op == "a":
                    n_add += 1
    return n_sub, n_del, n_add, n_total


def _wav_duration(wav: Path) -> float:
    try:
        info = sf.info(str(wav))
        return info.duration
    except Exception:
        return float("nan")


def main() -> None:
    if not L2_ROOT.exists():
        raise SystemExit(f"L2-ARCTIC root missing: {L2_ROOT}")

    rows: list[dict] = []
    per_rows: list[dict] = []

    for spk, (gender, l1) in SPEAKER_INFO.items():
        spk_dir = L2_ROOT / spk
        if not spk_dir.exists():
            logger.warning("missing speaker dir: %s", spk)
            continue
        wav_dir = spk_dir / "wav"
        tg_dir = spk_dir / "textgrid"
        ann_dir = spk_dir / "annotation"

        annot_ids = {p.stem for p in ann_dir.glob("*.TextGrid")} if ann_dir.exists() else set()

        # Speaker-level PER aggregation.
        spk_subs = spk_dels = spk_adds = spk_total = 0
        n_annot_files = 0

        for wav in sorted(wav_dir.glob("*.wav")):
            sid = wav.stem
            tg = tg_dir / f"{sid}.TextGrid"
            ann = ann_dir / f"{sid}.TextGrid"
            has_ann = sid in annot_ids
            dur = _wav_duration(wav)
            rows.append({
                "speaker_id": spk,
                "l1": l1,
                "gender": gender,
                "sentence_id": sid,
                "wav_path": str(wav.relative_to(PROJECT_ROOT)),
                "textgrid_path": str(tg.relative_to(PROJECT_ROOT)) if tg.exists() else "",
                "annotation_path": str(ann.relative_to(PROJECT_ROOT)) if has_ann else "",
                "has_annotation": has_ann,
                "duration_s": dur,
            })
            if has_ann:
                s, d, a, t = _parse_annotation(ann)
                spk_subs += s; spk_dels += d; spk_adds += a; spk_total += t
                n_annot_files += 1

        per = (spk_subs + spk_dels + spk_adds) / spk_total if spk_total > 0 else float("nan")
        per_rows.append({
            "speaker_id": spk, "l1": l1, "gender": gender,
            "n_annotated_utterances": n_annot_files,
            "n_subs": spk_subs, "n_dels": spk_dels, "n_adds": spk_adds,
            "n_phones_total": spk_total,
            "per": per,
        })
        logger.info("%-6s %-10s g=%s  utts=%d  annot=%d  PER=%.4f",
                    spk, l1, gender, sum(1 for r in rows if r["speaker_id"] == spk),
                    n_annot_files, per)

    md_df = pd.DataFrame(rows)
    per_df = pd.DataFrame(per_rows)
    md_out = PROJECT_ROOT / "data/processed/metadata/metadata_l2arctic.csv"
    per_out = PROJECT_ROOT / "results/l2arctic_speaker_per.csv"
    md_out.parent.mkdir(parents=True, exist_ok=True)
    per_out.parent.mkdir(parents=True, exist_ok=True)
    md_df.to_csv(md_out, index=False)
    per_df.to_csv(per_out, index=False)
    logger.info("wrote %s (%d rows)", md_out, len(md_df))
    logger.info("wrote %s (%d speakers)", per_out, len(per_df))
    print(per_df.to_string(index=False))


if __name__ == "__main__":
    main()
