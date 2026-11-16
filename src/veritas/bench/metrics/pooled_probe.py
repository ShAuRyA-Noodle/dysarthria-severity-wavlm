"""Reference metric #1 (probe): a fairly-tuned pooled-WavLM severity probe.

Ridge regression on frozen WavLM-Large layer-9 mean-pool features (speaker =
mean over utterances), with per-fold StandardScaler and nested alpha selection.
This is the project's "fair baseline" and reproduces the within-corpus success /
cross-corpus failure of Finding 1.

Doubles as a worked example of a **probe**-kind metric for the docs: it needs no
control reference, so the leakage test is N/A; it exposes the frozen-SSL
representation, so the transfer + identity-AUC probes apply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.veritas.bench.data import SpeakerView
from src.veritas.bench.metric import VeritasMetric
from src.veritas.bench.probe_common import within_loso_predict


class PooledWavLMProbe(VeritasMetric):
    name = "pooled_wavlm_probe"
    kind = "probe"
    has_representation = True
    representation_desc = "frozen WavLM-Large layer-9 mean-pool (1024-d), speaker=mean over utts"

    def _utt_frame(self, views: list[SpeakerView], feature_view: str):
        """Stack per-utterance features + metadata for the requested view."""
        Xs, rows = [], []
        for v in views:
            feats = v.pooled_full if feature_view == "full" else v.pooled_speech
            for u, x in zip(v.utt_ids, feats):
                Xs.append(x)
                rows.append({"utterance_id": u, "speaker_id": v.speaker_id,
                             "group": f"{v.corpus}::{v.speaker_id}",
                             "severity_rank": v.severity})
        X = np.stack(Xs, axis=0).astype(np.float64)
        meta = pd.DataFrame(rows)
        return X, meta

    def predictive_scores(
        self, views: list[SpeakerView], *, feature_view: str = "full"
    ) -> dict[str, float]:
        X, meta = self._utt_frame(views, feature_view)
        return within_loso_predict(X, meta)

    # probe does not leak via a control reference -> leakage test is N/A
    def loco_scores(self, views, *, feature_view: str = "full"):
        return None
