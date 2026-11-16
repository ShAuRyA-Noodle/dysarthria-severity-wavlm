"""VERITAS-Bench — a reusable VALIDITY test-suite for frozen-SSL clinical-speech metrics.

Turns the project's documented pitfalls (speaker leakage, silence, energy,
cross-corpus overfitting) into a battery any user can run against their OWN
per-speaker metric, emitting a machine-readable + human-readable validity
report card.

Quick start
-----------
    import src.veritas  # activates offline HF cache (must precede transformers)
    from src.veritas.bench import build_corpus_bundle, run_bench, render_table
    from src.veritas.bench.metric import VeritasMetric

    class MyMetric(VeritasMetric):
        name, kind = "my_metric", "intrinsic"
        def predictive_scores(self, views, *, feature_view="full"):
            return {v.speaker_id: my_score(v, feature_view) for v in views}

    bundle = build_corpus_bundle(records, device="cuda:0")
    card = run_bench(MyMetric(), bundle)
    print(render_table([card]))

See ``docs/veritas/VERITAS_BENCH.md`` for the full guide, thresholds, and the
reference results the shipped metrics produce on TORGO.
"""
from __future__ import annotations

import src.veritas as _veritas  # noqa: F401  (side effect: offline HF cache)

from src.veritas.bench.battery import run_bench
from src.veritas.bench.data import CorpusBundle, SpeakerView, build_corpus_bundle
from src.veritas.bench.metric import VeritasMetric
from src.veritas.bench.report import (
    ReportCard,
    TestResult,
    build_report,
    render_table,
    validate_card,
    validate_report,
    write_report,
)
from src.veritas.bench.thresholds import THRESHOLD_DOC, Outcome

__all__ = [
    "build_corpus_bundle",
    "CorpusBundle",
    "SpeakerView",
    "VeritasMetric",
    "run_bench",
    "ReportCard",
    "TestResult",
    "render_table",
    "build_report",
    "write_report",
    "validate_card",
    "validate_report",
    "Outcome",
    "THRESHOLD_DOC",
]
