"""VERITAS-Bench API.

A thin, honest HTTP API over the *actual* build in ``src/veritas/bench/``:

* ``GET  /api/report-cards`` serves the real committed report cards
  (``results/veritas/veritas_bench_report.json`` + the SOTA-audit card).
* ``POST /api/audit`` runs the **real** validity-threshold policy
  (``src/veritas/bench/thresholds.py``) live on user-supplied metric stats and
  returns the same PASS/WARN/FAIL verdict the benchmark would.

Run:
  C:/Users/Dell/.conda/envs/dysarthria/python.exe -m uvicorn backend.main:app --reload --port 8000
  (from the webapp/ directory)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "veritas"
THRESHOLDS_PY = REPO / "src" / "veritas" / "bench" / "thresholds.py"

# --- load the REAL threshold policy in isolation (thresholds.py only imports
#     enum + numpy, so no torch/transformers bootstrap is triggered) ----------
_spec = importlib.util.spec_from_file_location("veritas_thresholds", THRESHOLDS_PY)
TH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TH)  # type: ignore[union-attr]

_VALID = {o.value for o in TH.Outcome}

app = FastAPI(
    title="VERITAS-Bench API",
    version="1.0.0",
    description="Validity, not accuracy, for training-free clinical-speech metrics.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "policy_source": "src/veritas/bench/thresholds.py",
        "alpha": TH.ALPHA,
        "outcomes": sorted(_VALID),
    }


@app.get("/api/thresholds")
def thresholds():
    """The real, documented PASS/WARN/FAIL policy."""
    return TH.THRESHOLD_DOC


@app.get("/api/report-cards")
def report_cards():
    """Real committed report cards for all audited metrics on TORGO."""
    cards = []
    for f in ("veritas_bench_report.json", "veritas_bench_phonosubspace.json"):
        d = _load(f)
        if d:
            cards.extend(d.get("report_cards", []))
    return {"corpus": "TORGO", "seed": 42, "n_speakers": 15, "cards": cards}


@app.get("/api/summary")
def summary():
    """Headline numbers computed live from the committed report cards — so the
    frontend hard-codes nothing; every figure is derived from real results."""
    cards = report_cards()["cards"]

    def stat(card, ctrl, key):
        return ((card or {}).get("tests", {}).get(ctrl) or {}).get("stats", {}).get(key)

    rhos = [r for c in cards if (r := stat(c, "predictive_validity", "rho")) is not None]
    aucs = [a for c in cards if (a := stat(c, "leakage_by_layer", "speaker_identity_auc")) is not None]
    atd = next((c for c in cards if c.get("metric_name") == "atd_wavlm_l9"), None)
    sil = stat(atd, "silence_confound", "retention")
    return {
        "n_metrics": len(cards),
        "n_pass_all": sum(1 for c in cards if c.get("overall") == "PASS"),
        "rho_min": round(min(rhos), 2) if rhos else None,
        "rho_max": round(max(rhos), 2) if rhos else None,
        "auc_min": round(min(aucs), 3) if aucs else None,
        "auc_max": round(max(aucs), 3) if aucs else None,
        "atd_silence_lost_pct": round((1 - sil) * 100) if sil is not None else None,
        "atd_identity_auc": stat(atd, "leakage_by_layer", "speaker_identity_auc") if atd else None,
        "atd_transfer_outcome": ((atd or {}).get("tests", {}).get("cross_corpus_transfer") or {}).get("outcome"),
    }


@app.get("/api/findings")
def findings():
    """Curated real findings, each tracing to a committed result file."""
    return {
        "findings": [
            {"key": "silence", "title": "Pausing, not articulation",
             "stat": "24%", "detail": "Speech-only retention of the trajectory-distance "
             "signal; 76% of it lives in silence.", "source": "silence_confound.json"},
            {"key": "identity", "title": "The representation is the speaker",
             "stat": "0.996–0.999", "detail": "Speaker-classification AUC of every metric's "
             "representation (chance 0.5) — a systemic leakage risk.",
             "source": "veritas_bench_*.json"},
            {"key": "cycle", "title": "Cycle-consistency isn't reliability",
             "stat": "−0.001", "detail": "Energy-controlled partial Spearman vs measured EMA "
             "inversion error over 22,854 frames.", "source": "h1_faithfulness_vs_inversion_error.json"},
            {"key": "transfer", "title": "Transfer is metric-dependent",
             "stat": "n.s.", "detail": "Graded cross-corpus transfer is an underpowered null "
             "at n=15, not a proof of absence.", "source": "cross_corpus_severity.json"},
            {"key": "ema", "title": "The ground truth barely exists",
             "stat": "n≈2", "detail": "Usable dysarthric electromagnetic articulography sessions "
             "in public data.", "source": "h1_dysarthric_inversion.json"},
        ]
    }


class AuditIn(BaseModel):
    name: str = "my_metric"
    kind: str = "reference"  # reference | probe | intrinsic
    predictive_rho: float | None = None
    predictive_p: float | None = None
    leakage_delta: float | None = None
    silence_retention: float | None = None
    silence_fraction: float | None = None
    energy_retention: float | None = None
    transfer_rho: float | None = None
    transfer_p: float | None = None
    identity_auc: float | None = None


@app.post("/api/audit")
def audit(m: AuditIn):
    """Run the REAL threshold policy on supplied stats -> a live report card."""
    results = []

    def add(control, outcome, stat):
        results.append({"control": control, "outcome": outcome, "stat": stat})

    if m.predictive_rho is not None:
        o = TH.classify_predictive(m.predictive_rho, m.predictive_p if m.predictive_p is not None else 1.0)
        add("predictive_validity", o.value, f"rho={m.predictive_rho:g}")

    if m.kind == "reference" and m.leakage_delta is not None:
        add("leakage_loco", TH.classify_leakage(m.leakage_delta).value, f"delta={m.leakage_delta:g}")
    else:
        add("leakage_loco", "N/A", "not reference-based")

    if m.silence_retention is not None:
        o = TH.classify_silence(m.silence_retention, m.silence_fraction if m.silence_fraction is not None else 0.0)
        add("silence_confound", o.value, f"kept={m.silence_retention:g}")

    if m.energy_retention is not None:
        add("energy_confound", TH.classify_energy(m.energy_retention).value, f"kept={m.energy_retention:g}")

    if m.transfer_rho is not None:
        o = TH.classify_transfer(m.transfer_rho, m.transfer_p if m.transfer_p is not None else 1.0)
        add("cross_corpus_transfer", o.value, f"rho={m.transfer_rho:g}")

    if m.identity_auc is not None:
        add("leakage_by_layer", TH.classify_identity_auc(m.identity_auc).value, f"AUC={m.identity_auc:g}")

    outs = [TH.Outcome(r["outcome"]) for r in results if r["outcome"] in _VALID]
    overall = TH.worst(outs).value if outs else "N/A"
    return {"name": m.name, "kind": m.kind, "results": results, "overall": overall}
