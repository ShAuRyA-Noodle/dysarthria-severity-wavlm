#!/usr/bin/env bash
# Install Montreal Forced Aligner v3 + English ARPA models for Track B.
# Run once. Creates a separate conda env `aligner` (MFA's deps conflict
# with our `dysarthria` env).
set -euo pipefail

CONDA="/c/ProgramData/Anaconda3/Scripts/conda.exe"

"${CONDA}" create -n aligner -c conda-forge montreal-forced-aligner=3.0 -y
# shellcheck disable=SC1091
eval "$("${CONDA}" shell.bash hook)" && conda activate aligner

mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p        english_us_arpa

echo
echo "==> MFA installed in conda env 'aligner'."
echo "    Activate with: conda activate aligner"
echo "    Validate corpus: mfa validate <corpus_dir> english_us_arpa english_us_arpa"
echo "    Align:           mfa align    <corpus_dir> <dict.dict>   english_us_arpa <out_dir>"
