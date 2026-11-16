"""VERITAS-Bench — audit the STRONGEST current training-free metric family.

Runs the full validity battery on an independent reimplementation of the
phonological-subspace d' metric (Muller et al., 2026) on real TORGO, with
IPVS + EasyCall as cross-corpus transfer references.

Why this run matters
--------------------
That work controls token count and alignment quality thoroughly, and reports
robust cross-corpus / cross-lingual transfer. It does NOT report controlling for
two confounds this battery tests:

  * speaker-identity LEAKAGE — its contrast directions are estimated from the
    healthy-control pool, so a CONTROL speaker is scored against directions
    fitted partly on that speaker's own data. The LOCO test re-fits the
    directions excluding the test speaker and measures the inflation.
  * ENERGY — partial Spearman controlling for mean speech energy.

We report whatever comes out, in either direction. A metric that PASSES the
battery is a good result for the field, and for the battery.

NOTE: this is OUR reimplementation from the published description, not the
authors' code. Reported as such. Same frozen features + alignments as every
other metric here, so the comparison is like-for-like.

Run:
  C:/Users/Dell/.conda/envs/dysarthria/python.exe experiments/veritas/B4_audit_phonosubspace.py
Outputs:
  results/veritas/veritas_bench_phonosubspace.json
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
from src.veritas.bench.metrics import PhonologicalSubspace  # noqa: E402
from src.veritas.bench.report import build_report, write_report  # noqa: E402
from src.veritas.corpora import easycall, ipvs, torgo  # noqa: E402

SEED = 42
DEVICE = "cuda:0"
CAP = 80
OUT = PROJECT_ROOT / "results" / "veritas" / "veritas_bench_phonosubspace.json"


def main() -> int:
    set_global_seed(SEED)
    t0 = time.time()
    print("=" * 78)
    print(" VERITAS-BENCH  |  auditing the phonological-subspace d' metric family")
    print("=" * 78)

    torgo_recs = torgo.load()
    if not torgo_recs:
        raise RuntimeError("TORGO not available on disk")
    ipvs_recs = ipvs.load()
    easycall_recs = easycall.load()
    print(f"[load] torgo={len(torgo_recs)} ipvs={len(ipvs_recs)} easycall={len(easycall_recs)} utts")

    print("\n[bundle] torgo (all utterances) ...")
    torgo_full = build_corpus_bundle(torgo_recs, device=DEVICE, with_silence=True)

    other_bundles = []
    for recs, nm in [(ipvs_recs, "ipvs"), (easycall_recs, "easycall")]:
        if not recs:
            print(f"[bundle] {nm}: unavailable, skipping")
            continue
        print(f"[bundle] {nm} (capped, transfer reference) ...")
        other_bundles.append(build_corpus_bundle(
            recs, device=DEVICE, with_silence=False,
            max_utts_per_speaker=CAP, seed=SEED))

    metric = PhonologicalSubspace(layer=9)
    print("\n" + "=" * 78)
    print(f" full battery on {metric.name}  (reimplementation of Muller et al. 2026)")
    print("=" * 78)
    card = run_bench(metric, torgo_full, other_bundles=other_bundles, seed=SEED)
    card.notes.append(
        "Independent reimplementation from the published description, NOT the "
        "authors' code. Contrast directions estimated on healthy controls only; "
        "d' per speaker along those directions; score = -mean(d')."
    )
    card.notes.append(
        "Leakage test is meaningful here because the contrast directions are "
        "fitted on the control pool: the naive (as-published) form scores a "
        "control speaker against directions containing that speaker's own data."
    )

    config = {
        "seed": SEED,
        "device": DEVICE,
        "primary_corpus": "torgo",
        "reference_corpora": [b.name for b in other_bundles],
        "metric": metric.name,
        "metric_provenance": "REIMPLEMENTATION of Muller et al. 2026 "
                             "(arXiv:2604.10123 / 2604.21706) from the published "
                             "description; not the authors' code.",
        "contrasts": "nasality, voicing, stridency, sonorance, continuancy",
        "feature": "frozen WavLM-Large L9 phone-level embeddings via MFA alignment",
        "min_tokens_per_class": 5,
    }
    report = build_report([card], config)
    write_report(report, OUT)

    print("\n" + render_table([card]))
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
        OUT.with_name("veritas_bench_phonosubspace.error.json").write_text(
            json.dumps({"error": str(e), "traceback": tb}, indent=2), encoding="utf-8")
        sys.exit(1)
