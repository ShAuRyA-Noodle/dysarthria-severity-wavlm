"""Track B — Per-phoneme ATD pilot run on full TORGO corpus.

Requires:
  1. data/processed/textgrids/<speaker>/<utt_id>.TextGrid  (from MFA align)
  2. data/processed/references/reference_manifest.json       (TORGO refs)
  3. data/embeddings/wavlm_large/<speaker>/<utt_id>.npy      (SSL features)

Outputs:
  results/phoneme_atd.csv        — per-utterance, per-phoneme rows
  results/phoneme_norms.pkl      — z-norm distribution fit on controls
  results/phoneme_summary.json   — vowel-vs-consonant correlation summary
  results/figures/fig10_phoneme_heatmap.{png,pdf}

Run::

    python experiments/16_phoneme_pilot.py
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import LAYER_POSITION_MAP, RANDOM_SEED, TORGO_SEVERITY
from src.phoneme.per_phoneme_atd import manner_class, per_phoneme_atd
from src.phoneme.phoneme_norms import fit_norms_from_dataframe, save_norms, write_norms_json
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_reference_cache(manifest_path: Path, layer: int, project_root: Path) -> dict:
    """Pre-load per-word references for the target layer."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    cache: dict[str, dict] = {}
    for key, info in manifest.get("word_layer_ref", {}).items():
        if not key.endswith(f"_{layer}"):
            continue
        p = project_root / info["path"]
        if p.exists():
            with open(p, "rb") as rf:
                cache[key] = pickle.load(rf)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, process only this many utterances (debug).")
    parser.add_argument("--out", default="results/phoneme_atd.csv")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    layer_position = LAYER_POSITION_MAP[args.layer]
    layer = args.layer

    metadata_path = PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv"
    ssl_index_path = PROJECT_ROOT / "data/embeddings/ssl_index.json"
    refs_manifest_path = PROJECT_ROOT / "data/processed/references/reference_manifest.json"
    textgrid_root = PROJECT_ROOT / "data/processed/textgrids"

    logger.info("loading metadata + index + manifest...")
    metadata = pd.read_csv(metadata_path)
    with open(ssl_index_path, encoding="utf-8") as f:
        ssl_index = json.load(f)
    ref_cache = _load_reference_cache(refs_manifest_path, layer, PROJECT_ROOT)
    logger.info("ref cache: %d words loaded for layer %d", len(ref_cache), layer)

    started = time.time()
    rows: list[dict] = []
    n_aligned = 0
    n_no_textgrid = 0
    n_no_ref = 0
    n_no_index = 0
    n_no_npy = 0
    n_exc = 0
    first_exc: Exception | None = None
    iter_meta = metadata.itertuples(index=False)
    if args.limit > 0:
        iter_meta = list(iter_meta)[: args.limit]
    for r in iter_meta:
        utt_id = getattr(r, "utterance_id", None)
        spk = getattr(r, "speaker_id", None)
        word = getattr(r, "word", None)
        if not utt_id or spk not in TORGO_SEVERITY:
            continue
        tg_path = textgrid_root / spk / f"{utt_id}.TextGrid"
        if not tg_path.exists():
            n_no_textgrid += 1
            continue
        ref_key = f"{word}_{layer}"
        ref_data = ref_cache.get(ref_key)
        if ref_data is None:
            n_no_ref += 1
            continue
        entry = ssl_index.get(utt_id)
        if entry is None or entry.get("wavlm_large") is None:
            n_no_index += 1
            continue
        npy_path = PROJECT_ROOT / entry["wavlm_large"]
        if not npy_path.exists():
            n_no_npy += 1
            continue
        try:
            per_phone = per_phoneme_atd(
                utt_npy_path=npy_path,
                textgrid_path=tg_path,
                pca_model=ref_data["pca_model"],
                reference=ref_data["reference"],
                layer_position=layer_position,
            )
        except Exception as exc:  # noqa: BLE001
            n_exc += 1
            if first_exc is None:
                first_exc = exc
                logger.error("first per_phoneme_atd exception on %s: %r", utt_id, exc)
            continue
        sev = TORGO_SEVERITY[spk]
        is_control = sev == 0
        for p in per_phone:
            rows.append({
                "utterance_id": utt_id,
                "speaker_id": spk,
                "word": word,
                "severity_score": sev,
                "is_control": is_control,
                "layer": layer,
                **p,
                "manner": manner_class(p["phone"]),
            })
        n_aligned += 1
        if n_aligned % 500 == 0:
            logger.info("aligned %d utts; rows so far %d", n_aligned, len(rows))

    logger.info(
        "done. n_aligned=%d  rows=%d  no_textgrid=%d  no_ref=%d  no_index=%d  no_npy=%d  exc=%d  elapsed=%.1f min",
        n_aligned, len(rows), n_no_textgrid, n_no_ref, n_no_index, n_no_npy, n_exc,
        (time.time() - started) / 60,
    )

    df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("wrote %s rows=%d", out_csv, len(df))

    if df.empty:
        logger.warning("no rows generated; skipping norms + summary")
        return

    # Z-norm fit on controls only.
    norms = fit_norms_from_dataframe(df, is_control_field="is_control")
    norms_pkl = PROJECT_ROOT / "results/phoneme_norms.pkl"
    norms_json = PROJECT_ROOT / "results/phoneme_norms.json"
    save_norms(norms, norms_pkl)
    write_norms_json(norms, norms_json)
    logger.info("norms -> %s, %s", norms_pkl, norms_json)

    # Per-speaker manner-mean ATD vs severity (consonant vs vowel test).
    summary: dict = {"layer": layer, "n_utterances": int(n_aligned), "n_rows": int(len(df))}
    for manner_filter, label in [
        (lambda m: m == "vowel", "vowels_only"),
        (lambda m: m != "vowel" and m != "other", "consonants_only"),
        (lambda m: m == "voiceless_stop", "voiceless_stops_only"),
    ]:
        sub = df[df["manner"].apply(manner_filter)]
        spk_mean = sub.groupby("speaker_id")["atd"].mean()
        sev = pd.Series({s: TORGO_SEVERITY[s] for s in spk_mean.index})
        if len(spk_mean) >= 3:
            rho, p = spearmanr(sev.values, spk_mean.values)
            summary[label] = {
                "rho": float(rho), "p": float(p), "n_speakers": int(len(spk_mean)),
                "n_tokens": int(len(sub)),
            }
            logger.info("manner=%s rho=%.4f p=%.4f n_spk=%d n_tokens=%d",
                        label, rho, p, len(spk_mean), len(sub))

    summary_path = PROJECT_ROOT / "results/phoneme_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("summary -> %s", summary_path)


if __name__ == "__main__":
    main()
