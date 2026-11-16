import json
import subprocess
import sys

import torch

from src.constants import TORGO_SEVERITY


def test_v1_runs_smoke_with_failloud_accounting(tmp_path):
    out = tmp_path / "faith.json"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    r = subprocess.run(
        [
            sys.executable,
            "experiments/veritas/V1_faithfulness_vs_severity.py",
            "--limit-per-speaker", "2",
            "--mu-per-speaker", "3",
            "--device", device,
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())

    # correlation reported over all 15 TORGO speakers
    assert "spearman_rho" in d and "p_value" in d
    assert d["n_speakers"] == 15
    assert set(d["per_speaker"]) == set(TORGO_SEVERITY)

    # FIX 3: per-speaker fail-loud accounting
    for spk in TORGO_SEVERITY:
        e = d["per_speaker"][spk]
        for k in ("attempted", "used", "skipped"):
            assert k in e, f"{spk} missing accounting field {k}"
        assert e["used"] >= 1, f"{spk} produced 0 utterances"
        assert e["attempted"] == e["used"] + e["skipped"]

    # FIX 3: total skip count + coverage flag
    acc = d["accounting"]
    assert isinstance(acc["total_skipped"], int)
    assert acc["total_used"] == sum(d["per_speaker"][s]["used"] for s in TORGO_SEVERITY)
    assert acc["all_speakers_have_utterances"] is True

    # centered metric + P2 multi-layer report (primary layer 9 present)
    assert d["centered"] is True
    assert "9" in d["per_layer"]
    assert d["primary_layer"] == 9
