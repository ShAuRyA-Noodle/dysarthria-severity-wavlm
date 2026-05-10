"""End-to-end reproduction pipeline for ATD on TORGO + UASpeech.

Runs every canonical experiment in order. Each step is resumable: if its
expected outputs already exist on disk, it is skipped unless ``--force`` is
passed.

Usage
-----
    python reproduce_all.py                  # full TORGO pipeline, skip done
    python reproduce_all.py --force          # re-run every step
    python reproduce_all.py --skip-extract   # reuse existing embeddings
    python reproduce_all.py --include-uaspeech  # also run UA_01..UA_05
    python reproduce_all.py --smoke          # smoke-test (no full extraction)

Each step is shelled out as a separate process so a failure in step N does
not cripple the Python state — re-run from the same command. Logs are
written under ``results/logs/reproduce_all/<step>.log``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS = PROJECT_ROOT / "experiments"
RESULTS = PROJECT_ROOT / "results"
LOG_DIR = RESULTS / "logs" / "reproduce_all"


@dataclass
class Step:
    """One pipeline step: a script + the artifact whose presence means 'done'."""

    name: str
    script: str
    output_check: Path | None = None  # if exists & not --force, skip
    optional: bool = False  # if True, errors don't abort pipeline

    @property
    def script_path(self) -> Path:
        return EXPERIMENTS / self.script


TORGO_PIPELINE: list[Step] = [
    Step("01_verify_env",      "01_verify_environment.py",
         RESULTS / "logs" / "environment_check.txt"),
    Step("02_explore",         "02_explore_torgo.py",
         PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"),
    Step("03_preprocess",      "03_preprocess_audio.py",
         PROJECT_ROOT / "data" / "processed" / "audio_16khz"),
    Step("04_extract_ssl",     "04_extract_ssl.py",
         PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"),
    Step("05_build_refs",      "05_build_references.py",
         PROJECT_ROOT / "data" / "processed" / "references" / "reference_manifest.json"),
    Step("06_compute_atd",     "06_compute_atd.py",
         RESULTS / "atd_scores.csv"),
    Step("07_validate",        "07_validate_atd.py",
         RESULTS / "atd_validation_report.json"),
    Step("08_baselines",       "08_baselines.py",
         RESULTS / "tables" / "table1_main_results.csv"),
    Step("09_layer_analysis",  "09_layer_analysis.py",
         RESULTS / "tables" / "layer_analysis.csv"),
    Step("10_loso",            "10_loso_validation.py",
         RESULTS / "tables" / "loso_results.csv"),
    Step("11_ablations",       "11_ablations.py",
         RESULTS / "tables" / "table2_ablations.csv"),
    Step("12_figures",         "12_generate_figures.py",
         RESULTS / "figures" / "fig1_layer_analysis.pdf"),
    Step("13_tables",          "13_generate_tables.py",
         RESULTS / "paper_numbers.json"),
    Step("14_critical_fixes",  "14_critical_fixes.py",
         RESULTS / "critical_fixes_report.json"),
]

UASPEECH_PIPELINE: list[Step] = [
    Step("UA_01_explore",      "UA_01_explore_uaspeech.py",
         PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_uaspeech.csv",
         optional=True),
    Step("UA_02_preprocess",   "UA_02_preprocess_uaspeech.py", optional=True),
    Step("UA_03_extract_ssl",  "UA_03_extract_ssl_uaspeech.py", optional=True),
    Step("UA_04_atd",          "UA_04_cross_corpus_atd.py",
         RESULTS / "uaspeech_atd_scores.csv", optional=True),
    Step("UA_05_combined",     "UA_05_combined_analysis.py",
         RESULTS / "combined_validation_report.json", optional=True),
]


def _run_step(step: Step, *, force: bool, dry_run: bool, env_overrides: dict[str, str]) -> bool:
    """Run a single step. Return True on success (or skipped), False on hard fail."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step.name}.log"

    if not step.script_path.exists():
        msg = f"[SKIP] {step.name}: script not found at {step.script_path}"
        print(msg)
        if step.optional:
            return True
        return False

    if step.output_check is not None and step.output_check.exists() and not force:
        print(f"[SKIP] {step.name}: output exists ({step.output_check.name})")
        return True

    cmd = [sys.executable, str(step.script_path)]
    print(f"[RUN ] {step.name}: {' '.join(shlex.quote(c) for c in cmd)}")

    if dry_run:
        return True

    env = os.environ.copy()
    env.update(env_overrides)

    started = _dt.datetime.now()
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"# {step.name} | started {started.isoformat()}\n")
            logf.write(f"# cmd: {' '.join(cmd)}\n\n")
            logf.flush()
            result = subprocess.run(
                cmd, cwd=PROJECT_ROOT, env=env, stdout=logf,
                stderr=subprocess.STDOUT, check=False,
            )
    except KeyboardInterrupt:
        print(f"\n[ABRT] {step.name}: interrupted by user")
        raise

    elapsed = (_dt.datetime.now() - started).total_seconds()
    if result.returncode != 0:
        print(f"[FAIL] {step.name}: exit={result.returncode} after {elapsed:.1f}s — see {log_path}")
        return step.optional
    print(f"[ OK ] {step.name}: {elapsed:.1f}s")
    return True


def _summary(passed: list[str], failed: list[str], skipped: list[str]) -> None:
    print("\n" + "=" * 72)
    print(f"  reproduce_all summary — {_dt.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 72)
    print(f"  passed:  {len(passed)}  ({', '.join(passed) if passed else '—'})")
    print(f"  skipped: {len(skipped)} ({', '.join(skipped) if skipped else '—'})")
    print(f"  failed:  {len(failed)}  ({', '.join(failed) if failed else '—'})")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="Re-run every step regardless of existing outputs.")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip 04_extract_ssl (use cached embeddings).")
    parser.add_argument("--include-uaspeech", action="store_true",
                        help="Also run UA_01..UA_05 after the TORGO pipeline.")
    parser.add_argument("--smoke", action="store_true",
                        help="Dry-run: print step plan, don't execute.")
    parser.add_argument("--from-step", default=None,
                        help="Start from named step (e.g. 06_compute_atd).")
    parser.add_argument("--only", default=None,
                        help="Run only this single step (by name).")
    args = parser.parse_args(argv)

    pipeline = list(TORGO_PIPELINE)
    if args.include_uaspeech:
        pipeline.extend(UASPEECH_PIPELINE)
    if args.skip_extract:
        pipeline = [s for s in pipeline if s.name != "04_extract_ssl"]
    if args.from_step:
        names = [s.name for s in pipeline]
        if args.from_step not in names:
            print(f"--from-step {args.from_step!r} not in pipeline {names}")
            return 2
        pipeline = pipeline[names.index(args.from_step):]
    if args.only:
        pipeline = [s for s in pipeline if s.name == args.only]
        if not pipeline:
            print(f"--only {args.only!r} did not match any step")
            return 2

    print(f"Pipeline: {len(pipeline)} step(s)")
    for s in pipeline:
        print(f"  - {s.name}: {s.script}")
    print()

    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    env_overrides = {"PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "42"}

    for step in pipeline:
        out_existed = step.output_check is not None and step.output_check.exists() and not args.force
        ok = _run_step(step, force=args.force, dry_run=args.smoke, env_overrides=env_overrides)
        if not ok:
            failed.append(step.name)
            if not step.optional:
                _summary(passed, failed, skipped)
                return 1
            continue
        if out_existed:
            skipped.append(step.name)
        else:
            passed.append(step.name)

    _summary(passed, failed, skipped)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
