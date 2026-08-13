from collections import deque

import pytest
import torch

from anima_alive import (
    EPISTEMIC_CONTRADICTION,
    EPISTEMIC_FALSE,
    EPISTEMIC_TRUE,
    EPISTEMIC_UNDETERMINED,
    ConsciousMind,
)


class Cell:
    def __init__(self, cell_id, hidden, tension):
        self.id = cell_id
        self.hidden = torch.tensor([hidden], dtype=torch.float32)
        self.tension_history = [tension]


class Cells:
    def __init__(self, cells):
        self.cells = cells
        self._inter_tension_history = {}


def test_four_valued_epistemic_state_keeps_negation_unknown_and_conflict_distinct():
    classify = ConsciousMind.classify_epistemic_state
    assert classify(0.8, 0.2) == EPISTEMIC_TRUE
    assert classify(0.2, 0.8) == EPISTEMIC_FALSE
    assert classify(0.4, 0.4) == EPISTEMIC_UNDETERMINED
    assert classify(0.8, 0.8) == EPISTEMIC_CONTRADICTION


def test_one_shared_predictor_learns_self_and_other_and_lesions_selectively():
    torch.manual_seed(3)
    mind = ConsciousMind(dim=8, hidden=12)
    predictor_parameter_ids = {id(parameter) for parameter in mind.perspective_predictor.parameters()}

    for step in range(4):
        mind.observe_perspective(0.5 + step * 0.1, 0.2, 0.1, 'self', 'self')
        mind.observe_perspective(1.0 - step * 0.1, 0.4, 0.1, 'other', 'peer-a')

    state = mind.get_metacognition_state()
    assert state['perspective_metrics']['self']['samples'] == 3
    assert state['perspective_metrics']['other']['samples'] == 3
    assert {id(parameter) for parameter in mind.perspective_predictor.parameters()} == predictor_parameter_ids

    self_samples = state['perspective_metrics']['self']['samples']
    other_samples = state['perspective_metrics']['other']['samples']
    mind.set_pathology_intervention('self_perspective_enabled', False)
    disabled = mind.observe_perspective(0.9, 0.2, 0.1, 'self', 'self')
    mind.observe_perspective(0.8, 0.4, 0.1, 'other', 'peer-a')
    assert disabled['enabled'] is False
    assert mind._perspective_metrics['self']['samples'] == self_samples
    assert mind._perspective_metrics['other']['samples'] == other_samples + 1


def test_workspace_has_a_real_bottleneck_loser_trace_and_delayed_broadcast():
    mind = ConsciousMind(dim=4, hidden=4)
    cells = Cells([
        Cell('a', [1, 0, 0, 0], 0.9),
        Cell('b', [0, 1, 0, 0], 0.5),
        Cell('c', [0, 0, 1, 0], 0.2),
    ])
    summary = mind.update_global_workspace(cells)
    assert summary['winner_count'] == 1
    assert summary['winner_ids'] == ['a']
    assert summary['loser_trace_count'] == 2
    assert summary['broadcast_applied'] is True

    mind._contradiction_trace.append({
        'born_step': 0, 'strength': 0.8, 'remaining_frames': 3,
    })
    held = mind.update_global_workspace(cells)
    assert held['broadcast_applied'] is False
    assert held['loser_trace_count'] >= 1


def test_raw_label_loss_experience_frames_and_perspective_count_are_observable():
    torch.manual_seed(5)
    mind = ConsciousMind(dim=4, hidden=6)
    hidden = torch.zeros(1, 6)
    mind.set_pathology_intervention('experience_frame_steps', 2)

    for _ in range(3):
        output, _tension, _curiosity, direction, hidden = mind(torch.randn(1, 4), hidden)
        error = mind.observe_label_compression(direction, 0.2, 0.4, -0.1)
        assert 0.0 <= error <= 1.0

    frame = mind.get_experience_frame_summary()
    assert frame['completed_frames'] == 1
    assert frame['last_complete']['steps'] == 2
    assert frame['last_complete']['label_reconstruction_error'] >= 0.0

    cells = Cells([
        Cell('a', [1, 0, 0, 0, 0, 0], 0.5),
        Cell('b', [0.9, 0.1, 0, 0, 0, 0], 0.5),
        Cell('c', [-1, 0, 0, 0, 0, 0], 0.5),
    ])
    mind.update_metacognition(cells)
    assert mind.self_awareness['active_perspectives'] == 2


def test_introspection_feedback_only_changes_input_after_developmental_prerequisites():
    torch.manual_seed(7)
    mind = ConsciousMind(dim=8, hidden=10)
    hidden = torch.zeros(1, 10)
    mind.tension_history = [1.0] * 10
    mind._perspective_metrics['self'].update(samples=16, accuracy=0.8, brier=0.1)
    mind._perspective_metrics['other'].update(samples=8, accuracy=0.8, brier=0.1)
    mind._sensorimotor_closed_loop_samples = 8
    mind._sensorimotor_control_ema = 0.8
    mind._experience_frames = deque([
        {'integration': 0.5, 'phi': 0.1},
        {'integration': 0.6, 'phi': 0.1},
        {'integration': 0.7, 'phi': 0.1},
    ], maxlen=32)
    mind._recursive_self_observations = 8
    mind._introspection_feedback = torch.ones(1, 8)

    assert mind.get_development_state()['active_stage'] == 'recursive_self_model'
    with torch.no_grad():
        mind(torch.zeros(1, 8), hidden)
    assert mind._introspection_feedback_applied > 0

    mind._introspection_feedback = torch.ones(1, 8)
    mind.set_pathology_intervention('introspection_feedback_enabled', False)
    with torch.no_grad():
        mind(torch.zeros(1, 8), hidden)
    assert mind._introspection_feedback_applied == 0


def test_runtime_cognition_survives_checkpoint_round_trip():
    mind = ConsciousMind(dim=4, hidden=6)
    mind.observe_perspective(0.5, 0.2, 0.1, 'self', 'self')
    mind.observe_perspective(0.6, 0.3, 0.1, 'self', 'self')
    mind.observe_control_outcome('tool', 0.8, 1.0, 1.0)
    mind.observe_functional_cost(cost=0.1)
    state = mind.runtime_state_dict()

    restored = ConsciousMind(dim=4, hidden=6)
    assert restored.load_runtime_state_dict(state) is True
    assert restored._perspective_metrics['self']['samples'] == 1
    assert restored._self_boundary['tool'] == pytest.approx(mind._self_boundary['tool'])
    assert restored._functional_budget == pytest.approx(0.9)


def test_metric_read_does_not_create_an_experience_frame_sample():
    mind = ConsciousMind(dim=8)

    before = mind.get_experience_frame_summary()
    score = mind.get_consciousness_score()
    after = mind.get_experience_frame_summary()

    assert score['experience_frame'] == before
    assert after == before
    assert mind._open_experience_frame['phi_samples'] == []


def test_pathology_interventions_reject_invalid_targets_and_ranges():
    mind = ConsciousMind(dim=4, hidden=6)
    with pytest.raises(ValueError):
        mind.set_pathology_intervention('unknown', True)
    with pytest.raises(ValueError):
        mind.set_pathology_intervention('prediction_error_gain', 11)
    with pytest.raises(ValueError):
        mind.set_pathology_intervention('bottleneck_width', 0)
