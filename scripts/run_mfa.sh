#!/usr/bin/env bash
# Wrapper for MFA on Windows: prepends conda-env Library/bin + Scripts to
# PATH so OpenFST + Kaldi binaries resolve, then forwards args to mfa.exe.
#
# Usage::
#
#     scripts/run_mfa.sh validate <corpus_dir> <dict> <acoustic_model>
#     scripts/run_mfa.sh align    <corpus_dir> <dict> <acoustic_model> <out_dir>
#     scripts/run_mfa.sh model list dictionary
set -euo pipefail
ENV_ROOT="/c/Users/Dell/.conda/envs/aligner"
PATH="${ENV_ROOT}/Library/bin:${ENV_ROOT}/Scripts:${PATH}"
exec "${ENV_ROOT}/Scripts/mfa.exe" "$@"
