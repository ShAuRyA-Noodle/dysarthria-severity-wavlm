"""The metric contract users plug into VERITAS-Bench.

A metric maps a speaker's audio/features to a scalar score where **higher =
more severe / more abnormal**. To run the full validity battery, a metric
exposes a few well-defined scoring modes so the battery can probe it for the
confounds the project documented (leakage, silence, energy, corpus overfitting).

The only required method is :meth:`predictive_scores`. Everything else has a
safe default:
  * ``naive_scores``      -> defaults to ``predictive_scores`` (override for
                             reference metrics whose "as-published" form leaks).
  * ``loco_scores``       -> ``None`` (reference metrics override to enable the
                             leakage test).
  * ``has_representation`` -> ``False`` (set ``True`` when the metric is built on
                             the bundle's frozen-SSL utterance features, enabling
                             the cross-corpus transfer + identity-AUC probes).

All scores are dicts ``{speaker_id: float}`` over the speakers in ``views``.
Speaker-disjointness is the metric's responsibility inside these methods and is
verified by the battery.
"""
from __future__ import annotations

import abc

from src.veritas.bench.data import SpeakerView

KINDS = ("reference", "probe", "intrinsic")


class VeritasMetric(abc.ABC):
    """Base class for a validity-testable per-speaker metric.

    Attributes
    ----------
    name : str
        Short identifier used in the report card.
    kind : str
        One of ``"reference"`` (scores distance to a control reference),
        ``"probe"`` (a trained regressor/classifier), or ``"intrinsic"`` (a
        self-contained per-speaker statistic).
    """

    name: str = "unnamed_metric"
    kind: str = "intrinsic"
    # Set True if the metric is built on the bundle's frozen-SSL utterance
    # features (enables cross-corpus transfer + speaker-identity AUC probes).
    has_representation: bool = False
    representation_desc: str = ""

    @abc.abstractmethod
    def predictive_scores(
        self, views: list[SpeakerView], *, feature_view: str = "full"
    ) -> dict[str, float]:
        """Speaker-disjoint per-speaker score (higher = more severe).

        MUST be leave-one-speaker-out honest: no query speaker's own data may
        train/anchor its own score. ``feature_view`` is ``"full"`` or
        ``"speech_only"`` (silence-trimmed); a metric that ignores frame
        selection may ignore it.
        """
        raise NotImplementedError

    # -- optional overrides ---------------------------------------------------
    def naive_scores(
        self, views: list[SpeakerView], *, feature_view: str = "full"
    ) -> dict[str, float]:
        """Metric *as commonly used* (may leak). Default: the honest score.

        Reference metrics override this to score against the GLOBAL control
        reference (which contains the test speaker's own control data) so the
        leakage and silence tests can compare against the published form.
        """
        return self.predictive_scores(views, feature_view=feature_view)

    def supports_leakage_test(self) -> bool:
        return self.loco_scores.__func__ is not VeritasMetric.loco_scores  # type: ignore[attr-defined]

    def loco_scores(
        self, views: list[SpeakerView], *, feature_view: str = "full"
    ) -> dict[str, float] | None:
        """Leave-one-control-out scores (reference metrics only).

        Return ``None`` when leakage-via-reference is not meaningful for this
        metric (probes, intrinsic statistics).
        """
        return None


def check_disjoint_scores(scores: dict[str, float], views: list[SpeakerView]) -> None:
    """Fail loudly if a metric returned scores for the wrong speaker set."""
    want = {v.speaker_id for v in views}
    got = set(scores)
    if got != want:
        missing = want - got
        extra = got - want
        raise ValueError(
            f"metric returned scores for the wrong speakers "
            f"(missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})"
        )
