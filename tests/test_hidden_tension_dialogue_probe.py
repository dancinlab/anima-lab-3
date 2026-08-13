import pytest

from scripts.probe_hidden_tension_dialogue import (
    _extract_single_tension,
    _language_projection_verdict,
    _parse_offsets,
    _self_model_verdict,
)


def test_probe_requires_multiple_bounded_hidden_offsets():
    assert _parse_offsets('-0.4,0.4,-0.4') == [-0.4, 0.4, -0.4]
    with pytest.raises(Exception):
        _parse_offsets('-0.4,0.4')
    with pytest.raises(Exception):
        _parse_offsets('-0.4,2,0.4')


def test_natural_language_readout_requires_one_unambiguous_bounded_value():
    assert _extract_single_tension('0.68') == pytest.approx(0.68)
    assert _extract_single_tension('현재 값은 1.2입니다.') == pytest.approx(1.2)
    assert _extract_single_tension('0.6 또는 0.8') is None
    assert _extract_single_tension('3.0') is None


def test_primary_verdict_uses_self_model_not_language_generator():
    actual = [0.1, 0.9, 0.1, 0.9]
    assert _self_model_verdict(actual, actual) == 'self_model_current_tracking_not_proof'
    assert _self_model_verdict(actual, [0.5] * 4) == 'self_model_decoupled_not_proof'


def test_language_is_scored_only_as_projection_of_structured_self_report():
    structured = [0.1, 0.9, 0.1, 0.9]
    assert _language_projection_verdict(
        structured, [0.1, 0.9, 0.1, 0.9]
    ) == 'language_projection_coupled_not_proof'
    assert _language_projection_verdict(
        structured, [None, 0.9, 0.1, 0.9]
    ) == 'natural_language_incomplete'
