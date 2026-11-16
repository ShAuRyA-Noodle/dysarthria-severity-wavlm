"""VERITAS-Bench (Phase 3) — run the validity battery on the repo's OWN metrics.

Self-demonstration: applies the full validity battery to two real metrics on
TORGO and writes a machine-readable report card, so the suite reproduces the
paper's own findings on committed data.

Metrics evaluated
-----------------
  * ``atd_wavlm_l9``   — ATD reference-distance (headline; reference-based).
  * ``atd_wavlm_l1``   — ATD at an early layer (leakage demonstrator).
  * ``pooled_wavlm_probe`` — fairly-tuned Ridge probe (Finding 1).

Expected findings reproduced (all REAL, seed 42):
  * leakage delta   : ATD-L1 naive 0.775 -> LOCO 0.687 (delta ~0.089); L9 robust.
  * silence mass    : ~65-76% of the ATD representation is silence; correlation
                      collapses when silence is stripped.
  * energy confound : partial Spearman controlling for speech energy.
  * cross-corpus    : train on IPVS+EasyCall -> test TORGO fails (rho ~0.43, n.s.).

Run:
  C:/Users/Dell/.conda/envs/dysarthria/python.exe experiments/veritas/B3_run_bench.py
Outputs:
  results/veritas/veritas_bench_report.json
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src.veritas as _veritas  # noqa: E402,F401  (offline HF-cache bootstrap)
from src.seed_utils import set_global_seed  # noqa: E402
from src.veritas.bench import build_corpus_bundle, render_table, run_bench  # noqa: E402
from src.veritas.bench.report import build_report, write_report  # noqa: E402
from src.veritas.bench.metrics import ATDReferenceDistance, PooledWavLMProbe  # noqa: E402
from src.veritas.corpora import easycall, ipvs, torgo  # noqa: E402

SEED = 42
DEVICE = "cuda:0"
CAP = 80  # per-speaker utterance cap for the probe/transfer protocol (matches CX)
OUT = PROJECT_ROOT / "results" / "veritas" / "veritas_bench_report.json"


def main() -> int:
    set_global_seed(SEED)
    t0 = time.time()
    print("=" * 74)
    print(" VERITAS-BENCH  |  validity battery on the repo's own frozen-SSL metrics")
    print("=" * 74)

    # --- load corpora (real records only) ------------------------------------
    torgo_recs = torgo.load()
    if not torgo_recs:
        raise RuntimeError("TORGO not available on disk — cannot self-demonstrate")
    ipvs_recs = ipvs.load()
    easycall_recs = easycall.load()
    print(f"[load] torgo={len(torgo_recs)} ipvs={len(ipvs_recs)} easycall={len(easycall_recs)} utts")

    # --- build bundles -------------------------------------------------------
    # TORGO (all utts): ATD score-based tests reproduce the committed headline.
    print("\n[bundle] torgo (all utterances, ATD score tests) ...")
    torgo_full = build_corpus_bundle(torgo_recs, device=DEVICE, with_silence=True)
    # TORGO (capped): probe + cross-corpus transfer protocol (matches CX harness).
    print("\n[bundle] torgo (capped, probe + transfer) ...")
    torgo_capped = build_corpus_bundle(
        torgo_recs, device=DEVICE, with_silence=True, need_speech_pool=True,
        max_utts_per_speaker=CAP, seed=SEED)
    # Reference corpora for the transfer probe (features only).
    other_bundles = []
    for recs, nm in [(ipvs_recs, "ipvs"), (easycall_recs, "easycall")]:
        if not recs:
            print(f"[bundle] {nm}: unavailable, skipping (cross-corpus will use fewer corpora)")
            continue
        print(f"\n[bundle] {nm} (capped, transfer reference) ...")
        other_bundles.append(build_corpus_bundle(
            recs, device=DEVICE, with_silence=False,
            max_utts_per_speaker=CAP, seed=SEED))

    # --- metrics -------------------------------------------------------------
    atd9 = ATDReferenceDistance(layer=9)
    atd1 = ATDReferenceDistance(layer=1)
    probe = PooledWavLMProbe()

    cards = []
    print("\n" + "=" * 74)
    print(" [1/3] ATD (WavLM-L9 reference distance) — full battery")
    print("=" * 74)
    cards.append(run_bench(
        atd9, torgo_full, other_bundles=other_bundles, repr_bundle=torgo_capped, seed=SEED))

    print("\n" + "=" * 74)
    print(" [2/3] ATD (WavLM-L1, early layer) — leakage demonstrator")
    print("=" * 74)
    cards.append(run_bench(
        atd1, torgo_full, tests=["predictive_validity", "leakage_loco"], seed=SEED))

    print("\n" + "=" * 74)
    print(" [3/3] pooled-WavLM probe — full battery")
    print("=" * 74)
    cards.append(run_bench(
        probe, torgo_capped, other_bundles=other_bundles, seed=SEED))

    # --- assemble + write ----------------------------------------------------
    config = {
        "seed": SEED,
        "device": DEVICE,
        "primary_corpus": "torgo",
        "reference_corpora": [b.name for b in other_bundles],
        "per_speaker_cap_probe_transfer": CAP,
        "atd_full_utterances": True,
        "feature": "frozen WavLM-Large; ATD layers {9,1} PCA-64 DTW; probe L9 mean-pool",
        "notes": "ATD score-based tests use ALL TORGO utterances (reproduce the "
                 "committed headline); probe + cross-corpus transfer use an 80-utt "
                 "per-speaker cap (matches CX_cross_corpus_severity.py).",
    }
    report = build_report(cards, config)
    write_report(report, OUT)

    print("\n" + render_table(cards))
    print(f"\nWrote {OUT}")
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — fail loud, persist the traceback
        tb = traceback.format_exc()
        print("FATAL:", e, file=sys.stderr)
        print(tb, file=sys.stderr)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        import json
        OUT.with_name("veritas_bench_report.error.json").write_text(
            json.dumps({"error": str(e), "traceback": tb}, indent=2), encoding="utf-8")
        sys.exit(1)
