"""Reference metric adapters shipped with VERITAS-Bench.

These are REAL, reusable implementations of the repo's own metrics that double
as worked examples for plugging in your own:
  * :class:`PooledWavLMProbe`      — a probe-kind metric (Finding 1).
  * :class:`ATDReferenceDistance`  — a reference-kind metric (Findings 2 & 3).
"""
from src.veritas.bench.metrics.atd import ATDReferenceDistance
from src.veritas.bench.metrics.pooled_probe import PooledWavLMProbe

__all__ = ["PooledWavLMProbe", "ATDReferenceDistance"]
