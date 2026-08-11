import torch

from conjunction import ConjunctionEpisode
from cue_history import (
    _comparison, _event_counter, _history_variants, _reverse_events,
)
from measurement.cue_history_gate import _passes, _recovered, _sensitive
from measurement.cue_history_registry import CUE_HISTORY_SPEC, spec_sha256


def _episode(context: int, distractors=(0, 1)) -> ConjunctionEpisode:
    return ConjunctionEpisode(
        contexts=(context, (context + 1) % 8), keys=(0, 1), values=(2, 3),
        active_contexts=(context, (context + 1) % 8), active_keys=(0, 1),
        active_values=(2, 3), distractors=distractors, query_position=0,
    )


def test_registered_preregistration_and_digest_are_stable():
    assert CUE_HISTORY_SPEC["preregistration_commit"] == "f90ce65e5"
    assert len(spec_sha256()) == 64
    assert CUE_HISTORY_SPEC["thresholds"]["minimum_prediction_disagreement"] == 0.10


def test_reverse_events_preserves_query_and_event_multiset():
    source = _episode(2)
    changed = _reverse_events(source)
    assert changed.query_context == source.query_context
    assert changed.query_key == source.query_key
    assert changed.target == source.target
    assert _event_counter(changed) == _event_counter(source)
    assert changed.contexts == tuple(reversed(source.contexts))


def test_history_variants_swap_only_registered_parts():
    episodes = [_episode(0, (0, 1)), _episode(0, (2, 3))]
    variants = _history_variants(episodes)
    assert variants["distractor_swapped"][0].contexts == episodes[0].contexts
    assert variants["distractor_swapped"][0].distractors == episodes[1].distractors
    assert variants["both_changed"][0].distractors == episodes[1].distractors
    assert variants["both_changed"][0].contexts == tuple(reversed(episodes[0].contexts))


def test_identity_comparison_is_exact():
    states = torch.eye(3)
    labels = torch.tensor([0, 1, 2])
    predictions = labels.clone()
    row = _comparison(states, states, labels, predictions, predictions)
    assert row["prediction_agreement"] == 1.0
    assert row["prediction_disagreement"] == 0.0
    assert row["accuracy_gain"] == 0.0
    assert row["state_mse"] == 0.0


def _condition(accuracy: float, recall: float, disagreement: float = 0.0):
    return {
        "metric": {"accuracy": accuracy, "minimum_class_recall": recall},
        "comparison_to_original": {"prediction_disagreement": disagreement},
    }


def test_closed_recovery_and_sensitivity_thresholds():
    thresholds = CUE_HISTORY_SPEC["thresholds"]
    original = {
        "query_full": _condition(0.96, 0.89),
        "query_quarter_missing": _condition(0.89, 0.82),
    }
    changed = {
        "query_full": _condition(0.93, 0.80),
        "query_quarter_missing": _condition(0.92, 0.78, 0.12),
    }
    assert _passes(changed["query_full"]["metric"], thresholds)
    assert _recovered(changed, original, thresholds)
    assert _sensitive(changed, thresholds)
