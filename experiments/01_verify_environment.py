"""
Experiment 01 — Environment Verification
Checks Python, PyTorch, CUDA, GPU, packages, TORGO data, and disk space.
Writes full report to results/logs/environment_check.txt.
"""

import sys
import os
import platform
import importlib
import shutil
from pathlib import Path
from datetime import datetime


# ── Project root detection ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORGO_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "torgo"
RESULTS_LOG_DIR = PROJECT_ROOT / "results" / "logs"

# ── Expected packages (from requirements.txt) ──────────────────────────
REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("transformers", "transformers"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "sklearn"),
    ("scipy", "scipy"),
    ("fastdtw", "fastdtw"),
    ("umap", "umap"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("tqdm", "tqdm"),
    ("h5py", "h5py"),
    ("pyarrow", "pyarrow"),
    ("pytest", "pytest"),
    ("parselmouth", "parselmouth"),
    ("statsmodels", "statsmodels"),
]

# ── TORGO speaker directories ──────────────────────────────────────────
EXPECTED_SPEAKERS = {
    "F": ["F01", "F03", "F04"],
    "FC": ["FC01", "FC02", "FC03"],
    "M": ["M01", "M02", "M03", "M04", "M05"],
    "MC": ["MC01", "MC02", "MC03", "MC04"],
}


def check_python_version() -> tuple[bool, str]:
    """Check that Python version is 3.11.x."""
    ver = platform.python_version()
    ok = sys.version_info[:2] == (3, 11)
    msg = f"Python version: {ver}"
    if not ok:
        msg += f" — EXPECTED 3.11.x"
    return ok, msg


def check_pytorch_version() -> tuple[bool, str]:
    """Check that PyTorch has the patched version required by the project."""
    try:
        import torch
        ver = torch.__version__
        ok = ver.split("+")[0] == "2.13.0"
        msg = f"PyTorch version: {ver}"
        if not ok:
            msg += " — EXPECTED 2.13.0"
        return ok, msg
    except ImportError:
        return False, "PyTorch: NOT INSTALLED"


def check_cuda_available() -> tuple[bool, str]:
    """Check that CUDA is available."""
    try:
        import torch
        ok = torch.cuda.is_available()
        msg = f"CUDA available: {ok}"
        if not ok:
            msg += " — CUDA is required. Check your PyTorch installation."
        return ok, msg
    except ImportError:
        return False, "CUDA check: PyTorch not installed"


def check_gpu_name() -> tuple[bool, str]:
    """Check that GPU name contains '4080'."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "GPU name: CUDA not available"
        name = torch.cuda.get_device_name(0)
        ok = "4080" in name
        msg = f"GPU name: {name}"
        if not ok:
            msg += " — EXPECTED a GPU containing '4080'"
        return ok, msg
    except Exception as e:
        return False, f"GPU name check failed: {e}"


def check_gpu_vram() -> tuple[bool, str]:
    """Check that GPU VRAM is >= 10 GB."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "GPU VRAM: CUDA not available"
        total = torch.cuda.get_device_properties(0).total_memory
        total_gb = total / (1024 ** 3)
        ok = total_gb >= 10.0
        msg = f"GPU VRAM: {total_gb:.1f} GB"
        if not ok:
            msg += " — EXPECTED >= 10 GB"
        return ok, msg
    except Exception as e:
        return False, f"GPU VRAM check failed: {e}"


def check_packages() -> tuple[bool, str]:
    """Check that all required packages are importable."""
    missing = []
    versions = []
    for display_name, import_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "unknown")
            versions.append(f"  {display_name}: {ver}")
        except ImportError:
            missing.append(display_name)
            versions.append(f"  {display_name}: NOT FOUND")

    ok = len(missing) == 0
    header = f"Packages: {len(REQUIRED_PACKAGES) - len(missing)}/{len(REQUIRED_PACKAGES)} importable"
    if missing:
        header += f"\n  MISSING: {', '.join(missing)}"
        header += f"\n  FIX: pip install {' '.join(missing)}"
    detail = "\n".join(versions)
    return ok, f"{header}\n{detail}"


def check_torgo_data() -> tuple[bool, str]:
    """Check that TORGO data exists in data/raw/torgo/ with expected speaker dirs."""
    lines = []

    # Also check root-level directories as fallback
    root_torgo = PROJECT_ROOT
    torgo_dir = TORGO_DATA_DIR

    # Determine which path has the data
    use_dir = None
    if torgo_dir.exists():
        use_dir = torgo_dir
        lines.append(f"TORGO data directory: {torgo_dir} (exists)")
    elif all((root_torgo / group).exists() for group in EXPECTED_SPEAKERS):
        use_dir = root_torgo
        lines.append(f"TORGO data at project root (F/, FC/, M/, MC/ found)")
        lines.append(f"  NOTE: data/raw/torgo/ not found — using root-level dirs")
    else:
        lines.append(f"TORGO data: NOT FOUND at {torgo_dir} or project root")
        return False, "\n".join(lines)

    total_files = 0
    all_found = True
    for group, speakers in EXPECTED_SPEAKERS.items():
        group_dir = use_dir / group
        if not group_dir.exists():
            lines.append(f"  {group}/: MISSING")
            all_found = False
            continue
        for speaker in speakers:
            speaker_dir = group_dir / speaker
            if not speaker_dir.exists():
                lines.append(f"  {group}/{speaker}/: MISSING")
                all_found = False
                continue
            # Count all files recursively
            count = sum(1 for _ in speaker_dir.rglob("*") if _.is_file())
            total_files += count
            lines.append(f"  {group}/{speaker}/: {count:,} files")

    lines.append(f"  Total files across all speakers: {total_files:,}")
    return all_found, "\n".join(lines)


def check_disk_space() -> tuple[bool, str]:
    """Check that C: drive has >= 50 GB free."""
    try:
        usage = shutil.disk_usage("C:\\")
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        ok = free_gb >= 50.0
        msg = f"Disk space (C:): {free_gb:.1f} GB free / {total_gb:.1f} GB total"
        if not ok:
            msg += " — EXPECTED >= 50 GB free"
        return ok, msg
    except Exception as e:
        return False, f"Disk space check failed: {e}"


def main() -> None:
    """Run all environment checks and generate report."""
    print("=" * 70)
    print("  ATD Project — Environment Verification")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    checks = [
        ("1. Python version", check_python_version),
        ("2. PyTorch version", check_pytorch_version),
        ("3. CUDA available", check_cuda_available),
        ("4. GPU name", check_gpu_name),
        ("5. GPU VRAM", check_gpu_vram),
        ("6. Required packages", check_packages),
        ("7. TORGO data", check_torgo_data),
        ("8. Disk space", check_disk_space),
    ]

    results = []
    all_pass = True
    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append("  ATD Project — Environment Check Report")
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"  Project root: {PROJECT_ROOT}")
    report_lines.append("=" * 70)
    report_lines.append("")

    for label, fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        icon = "✅" if ok else "❌"
        results.append((label, ok, msg))

        print(f"{icon} [{status}] {label}")
        for line in msg.split("\n"):
            print(f"   {line}")
        print()

        report_lines.append(f"[{status}] {label}")
        for line in msg.split("\n"):
            report_lines.append(f"   {line}")
        report_lines.append("")

        if not ok:
            all_pass = False

    # Write report to file
    RESULTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_LOG_DIR / "environment_check.txt"
    report_lines.append("=" * 70)
    if all_pass:
        report_lines.append("RESULT: ALL CHECKS PASSED")
    else:
        failed = [label for label, ok, _ in results if not ok]
        report_lines.append(f"RESULT: {len(failed)} CHECK(S) FAILED")
        for f in failed:
            report_lines.append(f"  - {f}")
    report_lines.append("=" * 70)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"📄 Full report written to: {report_path}")
    print()

    # Final verdict
    print("=" * 70)
    if all_pass:
        print("✅ ENVIRONMENT READY — Proceed to Phase 1")
    else:
        print("❌ ENVIRONMENT NOT READY — Fix the following:")
        for label, ok, msg in results:
            if not ok:
                print(f"   • {label}")
                # Print just the first line of msg for the fix hint
                first_line = msg.split("\n")[0]
                print(f"     {first_line}")
    print("=" * 70)


if __name__ == "__main__":
    main()
