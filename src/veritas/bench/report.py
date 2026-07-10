"""Report-card data structures + schema validation + human-readable table.

A report card is deliberately plain (dataclasses -> dict -> JSON) so it is
machine-readable, diff-able in git, and trivially validated. One card per
(metric, corpus); a battery run over several metrics produces a list of cards
under a shared config header.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.veritas.bench.thresholds import Outcome, worst

SCHEMA_VERSION = "veritas-bench/1.0"

# Canonical order of the six tests on a card.
TEST_ORDER = [
    "predictive_validity",
    "leakage_loco",
    "silence_confound",
    "energy_confound",
    "cross_corpus_transfer",
    "leakage_by_layer",
]


@dataclass
class TestResult:
    """One test on one metric: a verdict, the headline stat, and raw details."""

    name: str
    outcome: str            # Outcome value
    headline: str           # one-line human summary with the load-bearing number
    stats: dict = field(default_factory=dict)   # raw numbers (JSON-safe)

    def to_dict(self) -> dict:
        return {"name": self.name, "outcome": self.outcome,
                "headline": self.headline, "stats": _jsonable(self.stats)}


@dataclass
class ReportCard:
    """Validity report card for one metric on one corpus."""

    metric_name: str
    metric_kind: str
    corpus: str
    n_speakers: int
    results: list[TestResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def overall(self) -> str:
        return worst([Outcome(r.outcome) for r in self.results]).value

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "metric_kind": self.metric_kind,
            "corpus": self.corpus,
            "n_speakers": int(self.n_speakers),
            "overall": self.overall,
            "tests": {r.name: r.to_dict() for r in self.results},
            "notes": list(self.notes),
        }


def _jsonable(obj):
    """Recursively coerce numpy / Path types into JSON-native ones."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --- schema validation --------------------------------------------------------
_REQUIRED_CARD_KEYS = {"metric_name", "metric_kind", "corpus", "n_speakers",
                       "overall", "tests", "notes"}
_REQUIRED_TEST_KEYS = {"name", "outcome", "headline", "stats"}
_VALID_OUTCOMES = {o.value for o in Outcome}


def validate_card(card: dict) -> list[str]:
    """Return a list of schema violations (empty list == valid)."""
    errs: list[str] = []
    missing = _REQUIRED_CARD_KEYS - set(card)
    if missing:
        errs.append(f"card missing keys: {sorted(missing)}")
    if card.get("overall") not in _VALID_OUTCOMES:
        errs.append(f"bad overall outcome: {card.get('overall')!r}")
    if not isinstance(card.get("n_speakers"), int):
        errs.append("n_speakers must be int")
    tests = card.get("tests", {})
    if not isinstance(tests, dict):
        errs.append("tests must be a dict")
        return errs
    for tname, t in tests.items():
        tmiss = _REQUIRED_TEST_KEYS - set(t)
        if tmiss:
            errs.append(f"test {tname!r} missing keys: {sorted(tmiss)}")
        if t.get("outcome") not in _VALID_OUTCOMES:
            errs.append(f"test {tname!r} bad outcome: {t.get('outcome')!r}")
    return errs


def validate_report(report: dict) -> list[str]:
    """Validate a full battery report (config header + list of cards)."""
    errs: list[str] = []
    if report.get("schema") != SCHEMA_VERSION:
        errs.append(f"schema must be {SCHEMA_VERSION!r}, got {report.get('schema')!r}")
    if "config" not in report:
        errs.append("missing config header")
    cards = report.get("report_cards")
    if not isinstance(cards, list) or not cards:
        errs.append("report_cards must be a non-empty list")
        return errs
    for c in cards:
        errs.extend(validate_card(c))
    return errs


# --- human-readable table -----------------------------------------------------
_SYMBOL = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "N/A": " -- "}


def render_table(cards: list[ReportCard]) -> str:
    """A compact, monospace validity table across metrics (rows = tests)."""
    lines: list[str] = []
    col_w = 22
    header = "TEST".ljust(28) + "".join(c.metric_name[:col_w].ljust(col_w) for c in cards)
    lines.append("=" * len(header))
    lines.append("VERITAS-BENCH validity report card")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    for tname in TEST_ORDER:
        row = tname.ljust(28)
        for c in cards:
            res = next((r for r in c.results if r.name == tname), None)
            cell = _SYMBOL.get(res.outcome, "?") if res else " -- "
            row += cell.ljust(col_w)
        lines.append(row)
    lines.append("-" * len(header))
    overall = "OVERALL".ljust(28) + "".join(_SYMBOL.get(c.overall, "?").ljust(col_w) for c in cards)
    lines.append(overall)
    lines.append("=" * len(header))
    # headline stat per metric/test
    for c in cards:
        lines.append("")
        lines.append(f"[{c.metric_name}]  ({c.metric_kind}, {c.corpus}, n={c.n_speakers})  ->  {c.overall}")
        for r in c.results:
            lines.append(f"  {_SYMBOL.get(r.outcome,'?')}  {r.name:24s} {r.headline}")
        for note in c.notes:
            lines.append(f"       note: {note}")
    return "\n".join(lines)


def build_report(cards: list[ReportCard], config: dict) -> dict:
    """Assemble the machine-readable battery report (validated on write)."""
    from src.veritas.bench.thresholds import THRESHOLD_DOC

    report = {
        "schema": SCHEMA_VERSION,
        "config": config,
        "thresholds": THRESHOLD_DOC,
        "report_cards": [c.to_dict() for c in cards],
    }
    errs = validate_report(report)
    if errs:
        raise ValueError("report failed schema validation: " + "; ".join(errs))
    return report


def write_report(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    tmp.replace(path)
