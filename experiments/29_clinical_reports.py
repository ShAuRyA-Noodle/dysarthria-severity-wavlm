"""Track 6 — LLM-driven clinical reports for TORGO dysarthric speakers.

Pipeline:
  1. Load results/phoneme_atd.csv + results/phoneme_norms.json.
  2. For each TORGO dysarthric speaker (severity >= 1):
       a. z-score every phoneme token vs the control distribution.
       b. Compute per-speaker top-K phonemes by |z|.
       c. Compute per-speaker mean z by manner class.
       d. Build the structured numeric brief.
       e. Call LLMClient (Gemini and Groq for cross-LLM consistency).
       f. Save report to results/clinical_reports/<speaker>.<backend>.md
          and a JSON metadata file with model + elapsed + numbers.

Rate-limited (Gemini 5 s/call, Groq 4 s/call). Cached on disk so re-runs
skip already-generated reports. Real LLM, real numbers in, real text out.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402

from src.clinical_reporter import LLMClient, SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from src.constants import TORGO_SEVERITY  # noqa: E402
from src.phoneme.per_phoneme_atd import manner_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    set_global_seed(42)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backends", default="gemini,groq")
    parser.add_argument("--top-k-phones", type=int, default=8)
    parser.add_argument("--include-controls", action="store_true",
                        help="Also generate reports for the 7 controls (baseline check).")
    args = parser.parse_args()

    phon_csv = PROJECT_ROOT / "results/phoneme_atd.csv"
    norms_json = PROJECT_ROOT / "results/phoneme_norms.json"
    if not phon_csv.exists() or not norms_json.exists():
        raise SystemExit("missing phoneme outputs — run 16_phoneme_pilot.py first")

    df = pd.read_csv(phon_csv)
    norms = json.loads(norms_json.read_text(encoding="utf-8"))
    phone_norms = {p: (v["mean"], v["std"]) for p, v in norms.get("by_phone", {}).items()}
    manner_norms = {m: (v["mean"], v["std"]) for m, v in norms.get("by_manner", {}).items()}

    def z(phone: str, atd: float) -> float:
        if phone in phone_norms:
            mu, sd = phone_norms[phone]
        else:
            mn = manner_class(phone)
            if mn not in manner_norms:
                return 0.0
            mu, sd = manner_norms[mn]
        if sd < 1e-9:
            return 0.0
        return (atd - mu) / sd

    df["z_atd"] = df.apply(lambda r: z(r["phone"], r["atd"]), axis=1)
    df["manner"] = df["phone"].map(manner_class)

    speaker_mean_atd = df.groupby("speaker_id")["atd"].mean()
    pop_mean = float(speaker_mean_atd.mean())
    pop_std = float(speaker_mean_atd.std(ddof=1))

    target_speakers = sorted(
        s for s, sev in TORGO_SEVERITY.items()
        if (sev >= 1 or args.include_controls)
    )
    logger.info("targets: %d speakers", len(target_speakers))

    clients = {}
    for backend in args.backends.split(","):
        backend = backend.strip()
        if not backend:
            continue
        try:
            min_iv = 5.0 if backend == "gemini" else 4.0
            clients[backend] = LLMClient(backend=backend, min_interval_s=min_iv)
        except Exception as exc:  # noqa: BLE001
            logger.error("init %s failed: %s", backend, exc)

    out_dir = PROJECT_ROOT / "results/clinical_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for spk in target_speakers:
        sub = df[df["speaker_id"] == spk]
        if sub.empty:
            continue
        phone_mean = (
            sub.groupby("phone")["z_atd"].mean().sort_values(key=lambda s: s.abs(), ascending=False)
        )
        top_phones = [(p, float(v)) for p, v in phone_mean.head(args.top_k_phones).items()]
        manner_mean = sub.groupby("manner")["z_atd"].mean().to_dict()
        manner_mean = {k: float(v) for k, v in manner_mean.items() if k != "other"}

        prompt_user = build_user_prompt(
            speaker_id=spk,
            severity_score=TORGO_SEVERITY[spk],
            speaker_mean_atd=float(speaker_mean_atd.loc[spk]),
            overall_population_mean=pop_mean,
            overall_population_std=pop_std,
            top_deviating_phonemes=top_phones,
            manner_class_summary=manner_mean,
        )

        for backend, client in clients.items():
            cache_key = f"{spk}_top{args.top_k_phones}_v1"
            try:
                r = client.chat(SYSTEM_PROMPT, prompt_user, cache_key=cache_key)
            except Exception as exc:  # noqa: BLE001
                logger.error("%s/%s: %s", spk, backend, exc)
                continue
            md_path = out_dir / f"{spk}.{backend}.md"
            md_path.write_text(
                f"# Clinical report — {spk} (backend={backend}, model={r.model})\n\n"
                f"## Input brief\n\n```\n{prompt_user}\n```\n\n"
                f"## LLM report\n\n{r.text}\n",
                encoding="utf-8",
            )
            manifest.append({
                "speaker_id": spk, "backend": backend, "model": r.model,
                "elapsed_s": r.elapsed_s, "cached": r.cached,
                "path": str(md_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            })
            logger.info("[%s/%s] %s (%.2fs, cached=%s)",
                        spk, backend, r.model, r.elapsed_s, r.cached)

    (out_dir / "manifest.json").write_text(
        json.dumps({"reports": manifest, "n_total": len(manifest)}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(manifest)} reports to {out_dir}")


if __name__ == "__main__":
    main()
