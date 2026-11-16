#!/usr/bin/env bash
# Install Montreal Forced Aligner v3 + English ARPA models for Track B.
#
# Tested 2026-05-10 on Windows 11 + Anaconda. The conda-forge MFA 3.0.7
# package alone does NOT work out of the box on Windows because:
#   1. joblib >=1.5 broke MFA's `Memory(bytes_limit=...)` API → pin to <1.5.
#   2. setuptools >=81 dropped pkg_resources → pin to <81 (praatio uses it).
#   3. soundfile sometimes ships a stub libsndfile.dll → force-reinstall.
#   4. The env's Library/bin (containing fstcompile.exe + Kaldi) is not on
#      PATH unless conda activates the env. Use the wrapper below.
#
# Run once. Creates a separate conda env `aligner` (MFA's deps conflict
# with our `dysarthria` env).
set -euo pipefail

CONDA="/c/ProgramData/Anaconda3/Scripts/conda.exe"
ENV="aligner"

"${CONDA}" create -n "${ENV}" -c conda-forge montreal-forced-aligner=3.0 -y

# Pinned-version fixes (must run in this order).
"${CONDA}" run -n "${ENV}" python -m pip install "joblib<1.5" "setuptools<81"
"${CONDA}" run -n "${ENV}" python -m pip install --force-reinstall --no-deps soundfile

# Verify mfa CLI loads.
PATH="/c/Users/Dell/.conda/envs/${ENV}/Library/bin:/c/Users/Dell/.conda/envs/${ENV}/Scripts:${PATH}" \
    /c/Users/Dell/.conda/envs/${ENV}/Scripts/mfa.exe version

# Download English ARPA models.
PATH="/c/Users/Dell/.conda/envs/${ENV}/Library/bin:/c/Users/Dell/.conda/envs/${ENV}/Scripts:${PATH}" \
    /c/Users/Dell/.conda/envs/${ENV}/Scripts/mfa.exe model download acoustic   english_us_arpa
PATH="/c/Users/Dell/.conda/envs/${ENV}/Library/bin:/c/Users/Dell/.conda/envs/${ENV}/Scripts:${PATH}" \
    /c/Users/Dell/.conda/envs/${ENV}/Scripts/mfa.exe model download dictionary english_us_arpa
PATH="/c/Users/Dell/.conda/envs/${ENV}/Library/bin:/c/Users/Dell/.conda/envs/${ENV}/Scripts:${PATH}" \
    /c/Users/Dell/.conda/envs/${ENV}/Scripts/mfa.exe model download g2p        english_us_arpa

echo
echo "==> MFA installed in conda env '${ENV}'."
echo "    To run mfa from anywhere, use the wrapper script:"
echo "        scripts/run_mfa.sh align <corpus_dir> <dict.dict> english_us_arpa <out_dir>"
echo "    The wrapper sets PATH so fstcompile.exe + Kaldi binaries resolve."
