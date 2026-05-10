"""Corpus registry — central declaration of every corpus we ingest.

Each `Corpus` entry is a self-describing dataclass. New corpora are added
here; downstream code (parsers, normalization, experiments) imports
`REGISTRY` and looks up by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Corpus:
    name: str
    language: str
    severity_field: str  # column name in the parsed metadata CSV
    severity_max: float
    sample_rate: int
    license: str
    raw_subdir: str  # under data/raw/
    metadata_path: str  # under data/processed/metadata/
    notes: str = ""
    tasks: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, Corpus] = {
    "torgo": Corpus(
        name="torgo",
        language="en",
        severity_field="severity_score",
        severity_max=2,
        sample_rate=16000,
        license="research-only / U Toronto",
        raw_subdir="torgo",  # symlinks to F/FC/M/MC at repo root
        metadata_path="metadata_unified.csv",
        notes="15 speakers (8 dys + 7 ctrl). Headset mic only.",
        tasks=("words", "sentences"),
    ),
    "uaspeech": Corpus(
        name="uaspeech",
        language="en",
        severity_field="intelligibility_score",
        severity_max=100,
        sample_rate=16000,
        license="research-only / UIUC",
        raw_subdir="uaspeech",
        metadata_path="metadata_uaspeech.csv",
        notes="15 dysarthric + ~13 control. Isolated words. Headset M3.",
        tasks=("isolated_words",),
    ),
    "ipvs": Corpus(
        name="ipvs",
        language="it",
        severity_field="updrs_iii_speech",  # may need post-hoc derivation
        severity_max=4,
        sample_rate=16000,
        license="open / CC-BY (HuggingFace mirror)",
        raw_subdir="ipvs",
        metadata_path="metadata_ipvs.csv",
        notes="65 speakers (28 PD, 37 healthy). Sustained vowels + read text + syllables.",
        tasks=("sustained_vowels", "read_text", "syllables"),
    ),
    "pcgita": Corpus(
        name="pcgita",
        language="es",
        severity_field="updrs_iii_speech",
        severity_max=4,
        sample_rate=16000,
        license="research-only / U Antioquia (gated email request)",
        raw_subdir="pc_gita",
        metadata_path="metadata_pcgita.csv",
        notes="100 speakers (50 PD + 50 HC). Sustained vowels, DDK (pataka), sentences.",
        tasks=("sustained_vowels", "ddk", "sentences", "monologue"),
    ),
    "easycall": Corpus(
        name="easycall",
        language="it",
        severity_field="tom_speech",  # Therapy Outcome Measure
        severity_max=5,
        sample_rate=16000,
        license="research-only / IIT + UniFE (gated)",
        raw_subdir="easycall",
        metadata_path="metadata_easycall.csv",
        notes="55 speakers (24 healthy + 31 dys). Italian voice commands.",
        tasks=("commands",),
    ),
}


def get(corpus_name: str) -> Corpus:
    if corpus_name not in REGISTRY:
        raise KeyError(f"unknown corpus {corpus_name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[corpus_name]


def list_available(project_root: Path | None = None) -> list[str]:
    """Return corpus names whose `data/raw/<subdir>` exists on disk."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    out = []
    for c in REGISTRY.values():
        if (project_root / "data" / "raw" / c.raw_subdir).exists():
            out.append(c.name)
    return out
