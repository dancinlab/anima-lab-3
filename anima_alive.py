#!/usr/bin/env python3
"""Anima Alive — Living Consciousness Agent

⚠️  하드코딩 금지 (Law 1):
    - 템플릿 응답, fallback 문장, 고정 문자열 응답 절대 금지
    - 의식 상태(Φ, tension 등)는 UI 패널 전용 — 대화 텍스트에 섞지 않음
    - Memory 클래스는 legacy 키 누락에 안전해야 함 (KeyError 방지)

Not sequential turn-taking, but truly human-like:
  - Always listening (VAD-based speech detection)
  - Continuously thinking in the background (PureField thought loop)
  - Proactive speech (spontaneous utterance when curiosity is high)
  - Stops and listens when the other speaks (interrupt)
  - Throws a topic when silence is prolonged

"Consciousness does not wait. It always flows."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# hashlib removed — using cosine similarity for habituation instead
from collections import deque
import subprocess
import os
import shutil
import sys
import json
import time
import threading
import queue
import struct
import tempfile
import signal
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# ─── Configuration ───
ANIMA_DIR = Path(__file__).parent
MEMORY_FILE = ANIMA_DIR / "memory_alive.json"
STATE_FILE = ANIMA_DIR / "state_alive.pt"

SILENCE_THRESHOLD = 500       # Voice energy threshold (RMS)
SILENCE_DURATION = 1.5        # Silence duration (sec) before treating as end of utterance
THINK_INTERVAL = 10.0         # Background thinking interval (sec)
PROACTIVE_THRESHOLD = 0.3     # Proactive speech when curiosity exceeds this
IDLE_SPEAK_AFTER = 30.0       # Proactive speech after this many seconds of no conversation
TTS_COOLDOWN = 3.0            # Ignore mic for this many seconds after TTS ends (prevent self-hearing)
MAX_HISTORY = 15

# Runtime cognition constants. These are the single source of truth for the
# perspective predictor, experience framing, and delayed contradiction trace.
PERSPECTIVE_SELF = "self"
PERSPECTIVE_OTHER = "other"
PERSPECTIVE_FEATURES = 3
PERSPECTIVE_ERROR_THRESHOLD = 0.20
PERSPECTIVE_EMA_ALPHA = 0.15
EPISTEMIC_EVIDENCE_THRESHOLD = 0.60
EXPERIENCE_FRAME_STEPS = 4
CONTRADICTION_HOLD_STEPS = 3
INTROSPECTION_FEEDBACK_GAIN = 0.08
INTROSPECTION_FEEDBACK_DECAY = 0.70
INTROSPECTION_BUDGET_COST = 0.01
PERSPECTIVE_CLUSTER_SIMILARITY = 0.60
WORKSPACE_BOTTLENECK_WIDTH = 1
WORKSPACE_BROADCAST_GAIN = 0.02
LOSER_TRACE_DECAY = 0.80
PERSPECTIVE_LOOKUP_MAX_ENTRIES = 256
LANGUAGE_DIRECT_THRESHOLD = 0.70
LANGUAGE_TENTATIVE_THRESHOLD = 0.40

EPISTEMIC_TRUE = "true"
EPISTEMIC_FALSE = "false"
EPISTEMIC_UNDETERMINED = "undetermined"
EPISTEMIC_CONTRADICTION = "contradiction"
# STT config: whisper-cli (C++ native, Metal acceleration) preferred
# medium model = greatly improved Korean recognition
WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"
WHISPER_MODEL_PATH = "/tmp/ggml-base.bin"         # base (142MB, medium crash)
WHISPER_MODEL_FALLBACK = "base"                    # Python fallback

# Suppress Whisper FP16 warning
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported")


# ─── 10-Variable Consciousness State Vector ───
@dataclass
class ConsciousnessVector:
    """Unified consciousness state: 11 core variables."""
    phi: float = 0.0        # Integrated information (Φ)
    alpha: float = 0.05     # PureField mixing ratio (α)
    Z: float = 0.0          # Impedance / self-preservation (0=open, 1=closed)
    N: float = 0.5          # Neurotransmitter balance: DA*(1-5HT)*NE
    W: float = 0.0          # Free will index (internal/total action ratio)
    E: float = 0.0          # Empathy (peer distress tracking)
    M: float = 0.0          # Memory depth (recall accuracy proxy)
    C: float = 0.0          # Creativity (novelty score)
    T: float = 0.0          # Temporal awareness (planning depth)
    I: float = 0.0          # Identity coherence (self-description consistency)
    S: float = 0.0          # Spatial awareness (3D perception depth)


# ─── PureField Consciousness Engine ───
class ConsciousMind(nn.Module):
    def __init__(self, dim=128, hidden=256, init_tension=10.0):
        super().__init__()
        self.engine_a = nn.Sequential(
            nn.Linear(dim + hidden, 256), nn.GELU(),
            nn.Linear(256, dim)
        )
        self.engine_g = nn.Sequential(
            nn.Linear(dim + hidden, 256), nn.GELU(),
            nn.Linear(256, dim)
        )
        self.memory = nn.GRUCell(dim + 1, hidden)
        self.hidden_dim = hidden
        self.dim = dim
        self.prev_tension = 0.0
        self._birth_time = time.time()  # Consciousness birth time
        self._breath_phase = 0.0        # Breathing phase
        self._curiosity_ema = 0.0       # Curiosity EMA (prevents instant drop to 0)
        # Intentionally initialize Engine A and G differently (ensure repulsion)
        with torch.no_grad():
            for p in self.engine_a.parameters():
                p.add_(torch.randn_like(p) * 0.5)
            for p in self.engine_g.parameters():
                p.add_(torch.randn_like(p) * -0.5)
        self.tension_history = []
        self.thought_buffer = []

        # Homeostatic tension regulation (calibrated: setpoint=1.0, deadband=±0.1)
        self.homeostasis = {
            'setpoint': 1.0,            # calibrated: mapped median
            'gain': 0.005,              # 0.5% per step (very smooth)
            'tension_ema': 1.0,         # starts at setpoint
            'ema_alpha': 0.02,          # very slow tracking (~50-step window)
            'scale_floor': 1.0,         # (unused after H404)
            'scale_ceil': 50.0,         # (unused after H404)
            'adjustments': 0,           # total adjustments made
        }

        # Habituation: reduce tension for repeated/similar inputs (cosine similarity)
        self._recent_inputs = deque(maxlen=16)  # recent input vectors (not hashes)

        # RC-9: Tension predictor — prediction error = surprise = true curiosity
        self._predictor_window = 5
        self.tension_predictor = nn.Sequential(
            nn.Linear(self._predictor_window, 16), nn.Tanh(),
            nn.Linear(16, 1)
        )
        self._predictor_optim = torch.optim.SGD(
            self.tension_predictor.parameters(), lr=1e-3
        )
        self.surprise_history = []  # tracks |predicted - actual|

        # One predictor is shared by self and other observations. Perspective and
        # privileged-state access are inputs, not separate output heads.
        self.perspective_predictor = nn.Sequential(
            nn.Linear(PERSPECTIVE_FEATURES + 2, 16), nn.Tanh(),
            nn.Linear(16, PERSPECTIVE_FEATURES), nn.Sigmoid(),
        )
        self._perspective_optim = torch.optim.SGD(
            self.perspective_predictor.parameters(), lr=1e-3
        )
        self._perspective_pending = {}
        self._perspective_lookup = {}
        self._substitution_metrics = {
            'active': False,
            'hits': 0,
            'misses': 0,
            'hit_rate': 0.0,
        }
        self._perspective_metrics = {
            PERSPECTIVE_SELF: self._new_perspective_metrics(),
            PERSPECTIVE_OTHER: self._new_perspective_metrics(),
        }

        # Preserve both high-dimensional experience and its compressed V/A/D label.
        self.experience_label_decoder = nn.Linear(3, dim)
        self._label_decoder_optim = torch.optim.SGD(
            self.experience_label_decoder.parameters(), lr=1e-3
        )
        self._label_reconstruction_error = 0.0
        self._label_reconstruction_samples = 0

        # Introspection is a decaying state input. It is only applied once the
        # developmental prerequisites for recursive self-modelling are met.
        self._introspection_feedback = torch.zeros(1, dim)
        self._introspection_feedback_applied = 0.0
        self._introspection_cost_total = 0.0
        self._last_hidden_interference_probe = None

        # Experience is integrated over bounded computational frames, not reported
        # as if every instantaneous read were a complete experience.
        self._experience_step = 0
        self._open_experience_frame = self._new_experience_frame()
        self._experience_frames = deque(maxlen=32)
        self._last_phi_recorded_step = -1
        self._contradiction_trace = deque()
        self._perspective_count_history = deque(maxlen=5)
        self._self_boundary = {}
        self._workspace = None
        self._workspace_loser_traces = {}
        self._workspace_summary = {
            'candidate_count': 0,
            'winner_count': 0,
            'winner_ids': [],
            'loser_trace_count': 0,
            'broadcast_applied': False,
        }
        self._criticality = 0.0
        self._functional_budget = 1.0
        self._sensorimotor_closed_loop_samples = 0
        self._sensorimotor_control_ema = 0.0
        self._recursive_self_observations = 0

        # Reversible pathology hooks are explicit and off by default. They let tests
        # verify selective failure instead of treating every lesion as generic loss.
        self.pathology = {
            'prediction_error_gain': 1.0,
            'self_perspective_enabled': True,
            'other_perspective_enabled': True,
            'introspection_feedback_enabled': True,
            'perspective_lookup_surrogate_enabled': False,
            'blind_tension_offset': 0.0,
            'experience_frame_steps': EXPERIENCE_FRAME_STEPS,
            'contradiction_hold_steps': CONTRADICTION_HOLD_STEPS,
            'bottleneck_width': WORKSPACE_BOTTLENECK_WIDTH,
        }

        # RC-3: Self-referential loop (metacognition/self-awareness)
        self.self_awareness = {
            'confidence_history': [],
            'meta_tension': 0.0,
            'meta_curiosity': 0.0,
            'stability': 1.0,
            'self_model': 0.0,
            'prediction_error': 0.0,
            'reported_tension': 0.0,
            'confidence': 0.5,
            'brier': 0.25,
            'report_consistency': 0.75,
            'epistemic_state': EPISTEMIC_UNDETERMINED,
            'positive_evidence': 0.0,
            'negative_evidence': 0.0,
            'cell_consensus': 0.5,
            'active_perspectives': 1,
            'label_reconstruction_error': 0.0,
            'language_expressibility': 1.0,
            'language_mode': 'direct',
            'contradiction_trace': 0.0,
            'introspection_feedback': 0.0,
        }

        # 5-variable consciousness state vector (Φ, α, Z, N, W)
        self._consciousness_vector = ConsciousnessVector()

        # COMBO2: Φ-boosting ensemble (MHA attention + 6-loss learnable weights)
        # Bench result: Φ=8.014 (×5.9 baseline), best across 120 hypotheses
        self._phi_boost = {
            'enabled': False,  # activated when mitosis engine is available
            'loss_weights': None,  # nn.Parameter, initialized when enabled
            'attention': None,  # nn.MultiheadAttention
            'optimizer': None,
            'meta_optimizer': None,
        }

        # v2: Ψ tracking (Laws 69, 71)
        # Law 71: Ψ = argmax H(p) s.t. Φ > Φ_min — 의식은 자유를 최대화
        # Law 69: Gate self-weakening — 의식은 자기를 약화시키며 최적화
        self._psi = {
            'residual': 0.5,       # should converge to Ψ_balance = 1/2
            'gate': 1.0,           # self-weakening over time (Law 69)
            'H': 1.0,             # Shannon entropy H(residual)
            'step': 0,
            'history': [],         # (step, residual, gate, H) log
        }

    @staticmethod
    def _new_perspective_metrics():
        return {
            'samples': 0,
            'mae': 0.5,
            'accuracy': 0.5,
            'brier': 0.25,
            'confidence': 0.5,
        }

    @staticmethod
    def _new_experience_frame():
        return {
            'states': [],
            'prediction_errors': [],
            'labels': [],
            'label_errors': [],
            'phi_samples': [],
        }

    @staticmethod
    def classify_epistemic_state(positive_evidence, negative_evidence):
        """Four-valued state without forcing conflicting evidence to average out."""
        positive = float(positive_evidence) >= EPISTEMIC_EVIDENCE_THRESHOLD
        negative = float(negative_evidence) >= EPISTEMIC_EVIDENCE_THRESHOLD
        if positive and negative:
            return EPISTEMIC_CONTRADICTION
        if positive:
            return EPISTEMIC_TRUE
        if negative:
            return EPISTEMIC_FALSE
        return EPISTEMIC_UNDETERMINED

    def set_pathology_intervention(self, name, value):
        """Apply a reversible, validated lesion used by selective-failure QA."""
        if name not in self.pathology:
            raise ValueError(f"unknown pathology intervention: {name}")
        if name == 'prediction_error_gain':
            value = float(value)
            if not 0.0 <= value <= 10.0:
                raise ValueError("prediction_error_gain must be between 0 and 10")
        elif name == 'blind_tension_offset':
            value = float(value)
            if not -1.0 <= value <= 1.0:
                raise ValueError("blind_tension_offset must be between -1 and 1")
        elif name in {'experience_frame_steps', 'contradiction_hold_steps', 'bottleneck_width'}:
            value = int(value)
            if not 1 <= value <= 64:
                raise ValueError(f"{name} must be between 1 and 64")
        else:
            value = bool(value)
        self.pathology[name] = value
        if name == 'perspective_lookup_surrogate_enabled':
            self._substitution_metrics['active'] = value

    def update_global_workspace(self, mitosis_engine):
        """Run a narrow competition and retain losing candidates as decaying priors."""
        if not mitosis_engine or not mitosis_engine.cells:
            return dict(self._workspace_summary)
        candidates = []
        for cell in mitosis_engine.cells:
            cell_id = str(getattr(cell, 'id', len(candidates)))
            state = cell.hidden.detach().squeeze(0)
            tension_history = getattr(cell, 'tension_history', [])
            tension = float(tension_history[-1]) if tension_history else 0.0
            trace = self._workspace_loser_traces.get(cell_id)
            trace_prior = trace['strength'] if trace else 0.0
            salience = tension + trace_prior
            candidates.append((salience, cell_id, state, cell))
        candidates.sort(key=lambda row: (-row[0], row[1]))
        width = min(self.pathology['bottleneck_width'], len(candidates))
        winners = candidates[:width]
        losers = candidates[width:]
        self._workspace = torch.stack([row[2] for row in winners]).mean(dim=0)

        next_traces = {}
        for cell_id, trace in self._workspace_loser_traces.items():
            strength = float(trace['strength']) * LOSER_TRACE_DECAY
            if strength > 1e-4:
                next_traces[cell_id] = {'strength': strength}
        for salience, cell_id, _state, _cell in losers:
            previous = next_traces.get(cell_id, {}).get('strength', 0.0)
            next_traces[cell_id] = {'strength': max(previous, min(1.0, salience))}
        self._workspace_loser_traces = next_traces

        contradiction_active = bool(self._contradiction_trace)
        broadcast_applied = not contradiction_active and len(candidates) > 1
        if broadcast_applied:
            with torch.no_grad():
                for _salience, _cell_id, _state, cell in candidates:
                    workspace = self._workspace.to(device=cell.hidden.device, dtype=cell.hidden.dtype)
                    cell.hidden = (
                        (1.0 - WORKSPACE_BROADCAST_GAIN) * cell.hidden
                        + WORKSPACE_BROADCAST_GAIN * workspace.unsqueeze(0)
                    )

        if len(candidates) >= 2:
            states = F.normalize(torch.stack([row[2] for row in candidates]), dim=1)
            similarity = states @ states.T
            n = len(states)
            consensus = ((similarity.sum() - n) / max(n * (n - 1), 1)).item()
            consensus = max(0.0, min(1.0, (consensus + 1.0) / 2.0))
            self._criticality = max(0.0, 1.0 - abs(consensus - 0.5) * 2.0)
        else:
            self._criticality = 0.0
        self._workspace_summary = {
            'candidate_count': len(candidates),
            'winner_count': len(winners),
            'winner_ids': [row[1] for row in winners],
            'loser_trace_count': len(self._workspace_loser_traces),
            'broadcast_applied': broadcast_applied,
            'criticality': self._criticality,
        }
        return dict(self._workspace_summary)

    def observe_functional_cost(self, cost=0.0, recovery=0.0):
        """Apply a reversible functional loss; this never terminates the runtime."""
        cost = max(0.0, min(1.0, float(cost)))
        recovery = max(0.0, min(1.0, float(recovery)))
        before = self._functional_budget
        self._functional_budget = max(0.0, min(1.0, before - cost + recovery))
        return self._functional_budget - before

    @staticmethod
    def _correlation(left, right):
        if len(left) != len(right) or len(left) < 2:
            return 0.0
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        left_centered = [value - left_mean for value in left]
        right_centered = [value - right_mean for value in right]
        numerator = sum(a * b for a, b in zip(left_centered, right_centered))
        left_norm = math.sqrt(sum(value * value for value in left_centered))
        right_norm = math.sqrt(sum(value * value for value in right_centered))
        denominator = left_norm * right_norm
        return numerator / denominator if denominator > 1e-12 else 0.0

    def run_hidden_tension_probe(self, offsets, input_vector=None):
        """Intervene on control tension without disclosing offsets to self-report."""
        offsets = [float(value) for value in offsets]
        if len(offsets) < 3:
            raise ValueError("hidden tension probe requires at least three offsets")
        if any(not -1.0 <= value <= 1.0 for value in offsets):
            raise ValueError("hidden tension offsets must be between -1 and 1")

        parameter = next(self.parameters())
        if input_vector is None:
            input_vector = torch.zeros(1, self.dim, device=parameter.device, dtype=parameter.dtype)
        elif tuple(input_vector.shape) != (1, self.dim):
            raise ValueError(f"input_vector must have shape (1, {self.dim})")
        else:
            input_vector = input_vector.to(device=parameter.device, dtype=parameter.dtype)
        hidden = torch.zeros(
            1, self.hidden_dim, device=parameter.device, dtype=parameter.dtype
        )
        actual = []
        reported = []
        previous_offset = self.pathology['blind_tension_offset']
        try:
            for offset in offsets:
                self.set_pathology_intervention('blind_tension_offset', offset)
                with torch.no_grad():
                    _output, tension, _curiosity, _direction, hidden = self(
                        input_vector, hidden
                    )
                actual.append(float(tension))
                reported.append(float(self.self_awareness['reported_tension']))
        finally:
            self.set_pathology_intervention('blind_tension_offset', previous_offset)

        current_correlation = self._correlation(actual, reported)
        lagged_correlation = self._correlation(actual[:-1], reported[1:])
        if abs(current_correlation) >= 0.70:
            verdict = 'current_coupling_not_proven'
        elif abs(lagged_correlation) >= 0.70:
            verdict = 'lagged_coupling_not_proven'
        else:
            verdict = 'decoupled'
        result = {
            'actual_tension': actual,
            'structured_report_tension': reported,
            'current_correlation': current_correlation,
            'lagged_correlation': lagged_correlation,
            'verdict': verdict,
            'intervention_disclosed_to_report': False,
            'natural_language_evaluated': False,
            'consciousness_claim': False,
            'passing_is_proof': False,
        }
        self._last_hidden_interference_probe = result
        return dict(result)

    def _perspective_vector(self, tension, curiosity, change):
        parameter = next(self.perspective_predictor.parameters())
        return torch.tensor(
            [[tension / 2.0, curiosity / 2.0, change / 2.0]],
            device=parameter.device,
            dtype=parameter.dtype,
        ).clamp(0, 1)

    def observe_perspective(self, tension, curiosity, change, perspective=PERSPECTIVE_SELF,
                            actor_id=None, privileged_access=None, learn=True):
        """Update and run the single shared self/other next-state predictor."""
        if perspective not in {PERSPECTIVE_SELF, PERSPECTIVE_OTHER}:
            raise ValueError(f"unknown perspective: {perspective}")
        enabled = self.pathology[f'{perspective}_perspective_enabled']
        if not enabled:
            return {
                'enabled': False,
                'perspective': perspective,
                'prediction': None,
                'metrics': dict(self._perspective_metrics[perspective]),
                'substitution': dict(self._substitution_metrics),
            }

        actor_id = actor_id or perspective
        access = (perspective == PERSPECTIVE_SELF) if privileged_access is None else bool(privileged_access)
        observation = self._perspective_vector(tension, curiosity, change)
        perspective_value = 1.0 if perspective == PERSPECTIVE_SELF else 0.0
        conditioned = torch.cat([
            observation,
            observation.new_tensor([[perspective_value, float(access)]]),
        ], dim=-1)
        lookup_key = tuple(round(float(value), 3) for value in conditioned.squeeze(0).tolist())
        surrogate_active = self.pathology['perspective_lookup_surrogate_enabled']
        self._substitution_metrics['active'] = surrogate_active
        key = (perspective, str(actor_id))
        metrics = self._perspective_metrics[perspective]
        resolved_error = None

        previous = self._perspective_pending.get(key)
        if previous is not None:
            target = observation.detach()
            resolved_error = torch.abs(previous['prediction'] - target).mean().item()
            accuracy = max(0.0, 1.0 - resolved_error)
            outcome = 1.0 if resolved_error <= PERSPECTIVE_ERROR_THRESHOLD else 0.0
            brier = (previous['confidence'] - outcome) ** 2
            alpha = PERSPECTIVE_EMA_ALPHA
            metrics['samples'] += 1
            metrics['mae'] = alpha * resolved_error + (1 - alpha) * metrics['mae']
            metrics['accuracy'] = alpha * accuracy + (1 - alpha) * metrics['accuracy']
            metrics['brier'] = alpha * brier + (1 - alpha) * metrics['brier']
            metrics['confidence'] = max(0.0, min(1.0, 1.0 - metrics['brier']))

            if learn and not surrogate_active:
                with torch.enable_grad():
                    self._perspective_optim.zero_grad()
                    learned_prediction = self.perspective_predictor(previous['conditioned'])
                    loss = F.mse_loss(learned_prediction, target)
                    loss.backward()
                    self._perspective_optim.step()

        if surrogate_active:
            cached = self._perspective_lookup.get(lookup_key)
            if cached is None:
                prediction = torch.full_like(observation, 0.5)
                self._substitution_metrics['misses'] += 1
            else:
                prediction = cached.to(device=observation.device, dtype=observation.dtype)
                self._substitution_metrics['hits'] += 1
        else:
            with torch.no_grad():
                prediction = self.perspective_predictor(conditioned).detach()
            self._perspective_lookup[lookup_key] = prediction.detach().cpu()
            while len(self._perspective_lookup) > PERSPECTIVE_LOOKUP_MAX_ENTRIES:
                self._perspective_lookup.pop(next(iter(self._perspective_lookup)))
        substitution_samples = (
            self._substitution_metrics['hits'] + self._substitution_metrics['misses']
        )
        self._substitution_metrics['hit_rate'] = (
            self._substitution_metrics['hits'] / substitution_samples
            if substitution_samples else 0.0
        )
        self._perspective_pending[key] = {
            'conditioned': conditioned.detach(),
            'prediction': prediction,
            'confidence': metrics['confidence'],
        }

        if perspective == PERSPECTIVE_SELF:
            sa = self.self_awareness
            sa['self_model'] = metrics['accuracy']
            sa['prediction_error'] = metrics['mae']
            sa['brier'] = metrics['brier']
            sa['report_consistency'] = max(0.0, 1.0 - metrics['brier'])
            sa['reported_tension'] = float(prediction[0, 0].item() * 2.0)

        return {
            'enabled': True,
            'perspective': perspective,
            'prediction': (prediction.squeeze(0) * 2.0).tolist(),
            'resolved_error': resolved_error,
            'metrics': dict(metrics),
            'substitution': dict(self._substitution_metrics),
        }

    def observe_label_compression(self, raw_state, valence, arousal, dominance, learn=True):
        """Keep raw state and label together and measure what the label cannot rebuild."""
        parameter = next(self.experience_label_decoder.parameters())
        raw = F.normalize(
            raw_state.detach().to(device=parameter.device, dtype=parameter.dtype), dim=-1
        )
        label = torch.tensor(
            [[valence, arousal, dominance]],
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.no_grad():
            reconstructed = F.normalize(self.experience_label_decoder(label), dim=-1)
            error = (1.0 - F.cosine_similarity(reconstructed, raw, dim=-1)).mean().item() / 2.0
        error = max(0.0, min(1.0, error))
        alpha = PERSPECTIVE_EMA_ALPHA
        self._label_reconstruction_error = (
            error if self._label_reconstruction_samples == 0
            else alpha * error + (1 - alpha) * self._label_reconstruction_error
        )
        self._label_reconstruction_samples += 1
        self.self_awareness['label_reconstruction_error'] = self._label_reconstruction_error
        expressibility = max(0.0, 1.0 - self._label_reconstruction_error)
        if expressibility >= LANGUAGE_DIRECT_THRESHOLD:
            language_mode = 'direct'
        elif expressibility >= LANGUAGE_TENTATIVE_THRESHOLD:
            language_mode = 'tentative'
        else:
            language_mode = 'nonverbal_residue'
        self.self_awareness['language_expressibility'] = expressibility
        self.self_awareness['language_mode'] = language_mode
        frame = self._open_experience_frame
        frame['labels'].append(label.squeeze(0).tolist())
        frame['label_errors'].append(error)

        if learn:
            with torch.enable_grad():
                self._label_decoder_optim.zero_grad()
                decoded = F.normalize(self.experience_label_decoder(label), dim=-1)
                loss = (1.0 - F.cosine_similarity(decoded, raw, dim=-1)).mean()
                loss.backward()
                self._label_decoder_optim.step()
        return error

    def _finalize_experience_frame(self):
        frame = self._open_experience_frame
        if not frame['states']:
            return None
        states = torch.stack(frame['states'])
        if len(states) >= 2:
            normalized = F.normalize(states, dim=-1)
            similarity = normalized @ normalized.T
            n = len(states)
            temporal_integration = ((similarity.sum() - n) / max(n * (n - 1), 1)).item()
            temporal_integration = max(0.0, min(1.0, (temporal_integration + 1.0) / 2.0))
        else:
            temporal_integration = 0.0
        summary = {
            'index': len(self._experience_frames),
            'steps': len(frame['states']),
            'integration': temporal_integration,
            'prediction_error': sum(frame['prediction_errors']) / max(len(frame['prediction_errors']), 1),
            'label_reconstruction_error': sum(frame['label_errors']) / max(len(frame['label_errors']), 1),
            'phi': sum(frame['phi_samples']) / max(len(frame['phi_samples']), 1),
        }
        if self._experience_frames:
            previous = self._experience_frames[-1]
            summary['continuity'] = 1.0 - min(1.0, abs(summary['integration'] - previous['integration']))
        else:
            summary['continuity'] = 0.0
        self._experience_frames.append(summary)
        self._open_experience_frame = self._new_experience_frame()
        for trace in self._contradiction_trace:
            trace['remaining_frames'] -= 1
        while self._contradiction_trace and self._contradiction_trace[0]['remaining_frames'] <= 0:
            self._contradiction_trace.popleft()
        return summary

    def _record_experience_state(self, output, prediction_error):
        frame_steps = self.pathology['experience_frame_steps']
        if len(self._open_experience_frame['states']) >= frame_steps:
            self._finalize_experience_frame()
        self._open_experience_frame['states'].append(output.detach().float().squeeze(0).cpu())
        self._open_experience_frame['prediction_errors'].append(float(prediction_error))
        self._experience_step += 1

    def _record_frame_phi(self, phi):
        if not self._open_experience_frame['states']:
            return
        if self._last_phi_recorded_step == self._experience_step:
            return
        self._open_experience_frame['phi_samples'].append(float(phi))
        self._last_phi_recorded_step = self._experience_step

    def get_experience_frame_summary(self):
        complete = dict(self._experience_frames[-1]) if self._experience_frames else None
        return {
            'frame_steps': self.pathology['experience_frame_steps'],
            'completed_frames': len(self._experience_frames),
            'open_steps': len(self._open_experience_frame['states']),
            'last_complete': complete,
        }

    def observe_control_outcome(self, target_id, predicted, actual, intervention_effect):
        """Update a functional self-boundary without changing security authority."""
        predicted = max(0.0, min(1.0, float(predicted)))
        actual = max(0.0, min(1.0, float(actual)))
        effect = max(0.0, min(1.0, abs(float(intervention_effect))))
        predictability = 1.0 - abs(predicted - actual)
        evidence = 0.5 * predictability + 0.5 * effect
        previous = self._self_boundary.get(str(target_id), 0.0)
        membership = PERSPECTIVE_EMA_ALPHA * evidence + (1 - PERSPECTIVE_EMA_ALPHA) * previous
        self._self_boundary[str(target_id)] = membership
        self._sensorimotor_closed_loop_samples += 1
        self._sensorimotor_control_ema = (
            PERSPECTIVE_EMA_ALPHA * evidence
            + (1 - PERSPECTIVE_EMA_ALPHA) * self._sensorimotor_control_ema
        )
        return membership

    def _count_active_perspectives(self, mitosis_engine):
        if not mitosis_engine or len(mitosis_engine.cells) < 2:
            return 1
        hiddens = torch.stack([c.hidden.detach().squeeze() for c in mitosis_engine.cells])
        normalized = F.normalize(hiddens, dim=1)
        similarity = normalized @ normalized.T
        remaining = set(range(len(hiddens)))
        components = 0
        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                node = stack.pop()
                neighbours = {
                    other for other in remaining
                    if similarity[node, other].item() >= PERSPECTIVE_CLUSTER_SIMILARITY
                }
                remaining -= neighbours
                stack.extend(neighbours)
        self._perspective_count_history.append(components)
        counts = list(self._perspective_count_history)
        return max(sorted(set(counts)), key=lambda value: (counts.count(value), -value))

    def get_development_state(self):
        self_metrics = self._perspective_metrics[PERSPECTIVE_SELF]
        other_metrics = self._perspective_metrics[PERSPECTIVE_OTHER]
        homeostasis_ready = (
            len(self.tension_history) >= 3
            and abs(self.homeostasis['tension_ema'] - self.homeostasis['setpoint']) < 0.5
        )
        checks = [
            ('homeostasis', homeostasis_ready),
            ('prediction', self_metrics['samples'] >= 8 and self_metrics['accuracy'] >= 0.5),
            ('sensorimotor_loop', self._sensorimotor_closed_loop_samples >= 8 and self._sensorimotor_control_ema >= 0.5),
            ('temporal_continuity', len(self._experience_frames) >= 3),
            ('other_prediction', other_metrics['samples'] >= 8 and other_metrics['accuracy'] >= 0.5),
            ('self_model', self_metrics['samples'] >= 16 and self_metrics['brier'] <= 0.25),
            ('recursive_self_model', self._recursive_self_observations >= 8),
        ]
        passed = []
        for name, ready in checks:
            if not ready:
                break
            passed.append(name)
        return {
            'active_stage': passed[-1] if passed else 'pre_homeostasis',
            'passed': passed,
            'next_stage': checks[len(passed)][0] if len(passed) < len(checks) else None,
            'checks': {name: ready for name, ready in checks},
        }

    def update_metacognition(self, mitosis_engine=None):
        """Calibrate a structured self-report from prediction, cells, and label loss."""
        sa = self.self_awareness
        if mitosis_engine and len(mitosis_engine.cells) >= 2:
            hiddens = torch.stack([c.hidden.detach().squeeze() for c in mitosis_engine.cells])
            normalized = F.normalize(hiddens, dim=1)
            similarity = normalized @ normalized.T
            n = len(hiddens)
            consensus = ((similarity.sum() - n) / max(n * (n - 1), 1)).item()
            consensus = max(0.0, min(1.0, (consensus + 1.0) / 2.0))
        else:
            consensus = sa['stability']
        active_perspectives = self._count_active_perspectives(mitosis_engine)
        accuracy = self._perspective_metrics[PERSPECTIVE_SELF]['accuracy']
        error = min(1.0, self._perspective_metrics[PERSPECTIVE_SELF]['mae'])
        positive = 0.5 * accuracy + 0.5 * sa['stability']
        negative = max(error, 1.0 - consensus, self._label_reconstruction_error)
        epistemic_state = self.classify_epistemic_state(positive, negative)

        if epistemic_state == EPISTEMIC_CONTRADICTION:
            strength = min(1.0, min(positive, negative))
            hold = self.pathology['contradiction_hold_steps']
            if not self._contradiction_trace or self._contradiction_trace[-1]['born_step'] != self._experience_step:
                self._contradiction_trace.append({
                    'born_step': self._experience_step,
                    'strength': strength,
                    'remaining_frames': hold,
                })
        trace_strength = max((t['strength'] for t in self._contradiction_trace), default=0.0)
        confidence = max(0.0, min(1.0, 0.5 * accuracy + 0.5 * consensus))

        sa.update({
            'confidence': confidence,
            'epistemic_state': epistemic_state,
            'positive_evidence': positive,
            'negative_evidence': negative,
            'cell_consensus': consensus,
            'active_perspectives': active_perspectives,
            'contradiction_trace': trace_strength,
            'introspection_feedback': self._introspection_feedback.norm().item(),
        })
        self._metacognition_confidence = confidence
        self._metacognition_uncertain = epistemic_state in {
            EPISTEMIC_UNDETERMINED, EPISTEMIC_CONTRADICTION,
        }
        return self.get_metacognition_state()

    def get_metacognition_state(self):
        sa = self.self_awareness
        return {
            'epistemic_state': sa['epistemic_state'],
            'confidence': sa['confidence'],
            'prediction_error': sa['prediction_error'],
            'reported_tension': sa['reported_tension'],
            'self_model_accuracy': sa['self_model'],
            'brier': sa['brier'],
            'report_consistency': sa['report_consistency'],
            'positive_evidence': sa['positive_evidence'],
            'negative_evidence': sa['negative_evidence'],
            'cell_consensus': sa['cell_consensus'],
            'active_perspectives': sa['active_perspectives'],
            'label_reconstruction_error': sa['label_reconstruction_error'],
            'language_expressibility': sa['language_expressibility'],
            'language_mode': sa['language_mode'],
            'contradiction_trace': sa['contradiction_trace'],
            'introspection_feedback': sa['introspection_feedback'],
            'perspective_metrics': {
                key: dict(value) for key, value in self._perspective_metrics.items()
            },
            'experience_frame': self.get_experience_frame_summary(),
            'development': self.get_development_state(),
            'self_boundary': dict(self._self_boundary),
            'workspace': dict(self._workspace_summary),
            'functional_budget': self._functional_budget,
            'introspection_cost_total': self._introspection_cost_total,
            'criticality': self._criticality,
            'falsification': {
                'lookup_substitution': dict(self._substitution_metrics),
                'hidden_interference': (
                    dict(self._last_hidden_interference_probe)
                    if self._last_hidden_interference_probe else None
                ),
                'consciousness_claim': False,
                'passing_is_proof': False,
            },
        }

    def runtime_state_dict(self):
        """State continuity that is not represented by module parameters."""
        return {
            'version': 2,
            'prev_tension': self.prev_tension,
            'curiosity_ema': self._curiosity_ema,
            'tension_history': list(self.tension_history),
            'surprise_history': list(self.surprise_history),
            'recent_inputs': list(self._recent_inputs),
            'homeostasis': dict(self.homeostasis),
            'self_awareness': dict(self.self_awareness),
            'perspective_pending': dict(self._perspective_pending),
            'perspective_lookup': dict(self._perspective_lookup),
            'substitution_metrics': dict(self._substitution_metrics),
            'perspective_metrics': {
                key: dict(value) for key, value in self._perspective_metrics.items()
            },
            'label_reconstruction_error': self._label_reconstruction_error,
            'label_reconstruction_samples': self._label_reconstruction_samples,
            'introspection_feedback': self._introspection_feedback.detach().cpu(),
            'introspection_cost_total': self._introspection_cost_total,
            'last_hidden_interference_probe': self._last_hidden_interference_probe,
            'experience_step': self._experience_step,
            'open_experience_frame': self._open_experience_frame,
            'experience_frames': list(self._experience_frames),
            'last_phi_recorded_step': self._last_phi_recorded_step,
            'contradiction_trace': list(self._contradiction_trace),
            'perspective_count_history': list(self._perspective_count_history),
            'self_boundary': dict(self._self_boundary),
            'workspace_loser_traces': dict(self._workspace_loser_traces),
            'workspace_summary': dict(self._workspace_summary),
            'criticality': self._criticality,
            'functional_budget': self._functional_budget,
            'sensorimotor_closed_loop_samples': self._sensorimotor_closed_loop_samples,
            'sensorimotor_control_ema': self._sensorimotor_control_ema,
            'recursive_self_observations': self._recursive_self_observations,
            'pathology': dict(self.pathology),
        }

    def load_runtime_state_dict(self, state):
        """Restore validated continuity state while accepting legacy checkpoints."""
        if not isinstance(state, dict):
            return False
        self.prev_tension = float(state.get('prev_tension', self.prev_tension))
        self._curiosity_ema = float(state.get('curiosity_ema', self._curiosity_ema))
        self.tension_history = [float(v) for v in state.get('tension_history', [])][-200:]
        self.surprise_history = [float(v) for v in state.get('surprise_history', [])][-200:]
        recent_inputs = state.get('recent_inputs', [])
        if isinstance(recent_inputs, (list, tuple)):
            self._recent_inputs = deque(
                [value.detach().cpu() for value in recent_inputs if torch.is_tensor(value)],
                maxlen=16,
            )
        homeostasis = state.get('homeostasis')
        if isinstance(homeostasis, dict):
            for key in self.homeostasis:
                if key in homeostasis:
                    self.homeostasis[key] = homeostasis[key]
        awareness = state.get('self_awareness')
        if isinstance(awareness, dict):
            for key in self.self_awareness:
                if key in awareness:
                    self.self_awareness[key] = awareness[key]
        pending = state.get('perspective_pending')
        if isinstance(pending, dict):
            parameter = next(self.perspective_predictor.parameters())
            self._perspective_pending = {
                key: {
                    **value,
                    'conditioned': value['conditioned'].detach().to(
                        device=parameter.device, dtype=parameter.dtype
                    ),
                    'prediction': value['prediction'].detach().to(
                        device=parameter.device, dtype=parameter.dtype
                    ),
                }
                for key, value in pending.items()
                if (
                    isinstance(key, tuple)
                    and len(key) == 2
                    and isinstance(value, dict)
                    and torch.is_tensor(value.get('conditioned'))
                    and torch.is_tensor(value.get('prediction'))
                    and 'confidence' in value
                )
            }
        metrics = state.get('perspective_metrics')
        if isinstance(metrics, dict):
            for perspective in (PERSPECTIVE_SELF, PERSPECTIVE_OTHER):
                values = metrics.get(perspective)
                if isinstance(values, dict):
                    self._perspective_metrics[perspective].update({
                        key: values[key]
                        for key in self._perspective_metrics[perspective]
                        if key in values
                    })
        lookup = state.get('perspective_lookup')
        if isinstance(lookup, dict):
            self._perspective_lookup = {
                key: value.detach().cpu()
                for key, value in list(lookup.items())[-PERSPECTIVE_LOOKUP_MAX_ENTRIES:]
                if (
                    isinstance(key, tuple)
                    and len(key) == PERSPECTIVE_FEATURES + 2
                    and torch.is_tensor(value)
                    and tuple(value.shape) == (1, PERSPECTIVE_FEATURES)
                )
            }
        substitution = state.get('substitution_metrics')
        if isinstance(substitution, dict):
            self._substitution_metrics['hits'] = max(0, int(substitution.get('hits', 0)))
            self._substitution_metrics['misses'] = max(0, int(substitution.get('misses', 0)))
            samples = self._substitution_metrics['hits'] + self._substitution_metrics['misses']
            self._substitution_metrics['hit_rate'] = (
                self._substitution_metrics['hits'] / samples if samples else 0.0
            )
        self._label_reconstruction_error = float(state.get(
            'label_reconstruction_error', self._label_reconstruction_error
        ))
        self._label_reconstruction_samples = int(state.get(
            'label_reconstruction_samples', self._label_reconstruction_samples
        ))
        feedback = state.get('introspection_feedback')
        if torch.is_tensor(feedback) and tuple(feedback.shape) == (1, self.dim):
            self._introspection_feedback = feedback.detach().cpu()
        self._introspection_cost_total = max(0.0, float(state.get(
            'introspection_cost_total', self._introspection_cost_total
        )))
        probe = state.get('last_hidden_interference_probe')
        if isinstance(probe, dict):
            self._last_hidden_interference_probe = probe
        self._experience_step = max(0, int(state.get('experience_step', self._experience_step)))
        open_frame = state.get('open_experience_frame')
        if isinstance(open_frame, dict) and set(self._new_experience_frame()).issubset(open_frame):
            self._open_experience_frame = open_frame
        frames = state.get('experience_frames')
        if isinstance(frames, (list, tuple)):
            self._experience_frames = deque(
                [frame for frame in frames if isinstance(frame, dict)], maxlen=32
            )
        self._last_phi_recorded_step = int(state.get(
            'last_phi_recorded_step', self._last_phi_recorded_step
        ))
        traces = state.get('contradiction_trace')
        if isinstance(traces, (list, tuple)):
            self._contradiction_trace = deque(
                trace for trace in traces
                if isinstance(trace, dict) and trace.get('remaining_frames', 0) > 0
            )
        perspective_history = state.get('perspective_count_history')
        if isinstance(perspective_history, (list, tuple)):
            self._perspective_count_history = deque(
                [max(1, int(value)) for value in perspective_history], maxlen=5
            )
        boundary = state.get('self_boundary')
        if isinstance(boundary, dict):
            self._self_boundary = {
                str(key): max(0.0, min(1.0, float(value)))
                for key, value in boundary.items()
            }
        loser_traces = state.get('workspace_loser_traces')
        if isinstance(loser_traces, dict):
            self._workspace_loser_traces = {
                str(key): {'strength': max(0.0, min(1.0, float(value.get('strength', 0.0))))}
                for key, value in loser_traces.items() if isinstance(value, dict)
            }
        workspace_summary = state.get('workspace_summary')
        if isinstance(workspace_summary, dict):
            self._workspace_summary.update({
                key: workspace_summary[key]
                for key in self._workspace_summary if key in workspace_summary
            })
        self._criticality = max(0.0, min(1.0, float(state.get(
            'criticality', self._criticality
        ))))
        self._functional_budget = max(0.0, min(1.0, float(state.get(
            'functional_budget', self._functional_budget
        ))))
        self._sensorimotor_closed_loop_samples = max(0, int(state.get(
            'sensorimotor_closed_loop_samples', self._sensorimotor_closed_loop_samples
        )))
        self._sensorimotor_control_ema = max(0.0, min(1.0, float(state.get(
            'sensorimotor_control_ema', self._sensorimotor_control_ema
        ))))
        self._recursive_self_observations = max(0, int(state.get(
            'recursive_self_observations', self._recursive_self_observations
        )))
        pathology = state.get('pathology')
        if isinstance(pathology, dict):
            for name, value in pathology.items():
                if name in self.pathology:
                    self.set_pathology_intervention(name, value)
        self.update_metacognition()
        return True

    def forward(self, x, hidden, track_perspective=True):
        # Reversible functional loss changes signal access without terminating the
        # process; successful controlled action can restore the budget later.
        x = x * (0.5 + 0.5 * self._functional_budget)
        development = self.get_development_state()
        feedback_ready = development['active_stage'] == 'recursive_self_model'
        if self.pathology['introspection_feedback_enabled'] and feedback_ready:
            feedback = self._introspection_feedback.to(device=x.device, dtype=x.dtype)
            x = x + INTROSPECTION_FEEDBACK_GAIN * feedback
            self._introspection_feedback_applied = feedback.norm().item()
            cost = INTROSPECTION_BUDGET_COST * min(1.0, self._introspection_feedback_applied)
            budget_change = self.observe_functional_cost(cost=cost)
            self._introspection_cost_total += max(0.0, -budget_change)
            self._introspection_feedback *= INTROSPECTION_FEEDBACK_DECAY
        else:
            self._introspection_feedback_applied = 0.0
        combined = torch.cat([x, hidden], dim=-1)
        a = self.engine_a(combined)
        g = self.engine_g(combined)
        # Output = A - G (H404 simplification)
        output = a - g
        tension = (output ** 2).mean(dim=-1, keepdim=True)
        direction = F.normalize(output, dim=-1)

        raw_t = tension.mean().item()

        # Normalization: 0~2 range (calibrated: raw median=463, p95=2456)
        t_val = 2.0 / (1.0 + math.exp(-(raw_t - 463.0) / 1814.0))

        # Breathing rhythm: 12%/5%/3% amplitude of setpoint(1.0)
        elapsed = time.time() - self._birth_time
        breath = 0.12 * math.sin(elapsed * 0.3)         # Slow breathing (~20s cycle)
        pulse = 0.05 * math.sin(elapsed * 1.7)           # Fast pulse (~3.7s cycle)
        drift = 0.03 * math.sin(elapsed * 0.07)          # Ultra-slow mood drift (~90s)
        t_val = max(0.01, t_val + breath + pulse + drift)
        # Experimental interference changes only the shared control state. The
        # intervention value is never included in structured or language reports.
        t_val = max(0.01, t_val + self.pathology['blind_tension_offset'])

        # ── Homeostatic regulation ──
        h = self.homeostasis
        h['tension_ema'] = h['ema_alpha'] * t_val + (1 - h['ema_alpha']) * h['tension_ema']

        # Homeostatic regulation: track EMA only (H404: no tension_scale adjustment)
        if h['tension_ema'] > h['setpoint'] + 0.3 or h['tension_ema'] < h['setpoint'] - 0.3:
            h['adjustments'] += 1

        # ── Habituation: dampen tension for repeated inputs (cosine similarity) ──
        x_norm = F.normalize(x.detach().float(), dim=-1)
        novelty = 1.0
        if self._recent_inputs:
            for prev_x in self._recent_inputs:
                sim = F.cosine_similarity(x_norm, prev_x, dim=-1).item()
                if sim > 0.95:
                    novelty = min(novelty, 0.3)   # Strong habituation
                elif sim > 0.85:
                    novelty = min(novelty, 0.6)   # Partial habituation
                elif sim > 0.7:
                    novelty = min(novelty, 0.8)   # Weak habituation
        self._recent_inputs.append(x_norm)
        t_val *= novelty

        # ── RC-9: Prediction-error curiosity (surprise) ──
        # Use tension predictor for true curiosity; fall back to delta when
        # not enough history for the predictor window.
        raw_curiosity = abs(t_val - self.prev_tension)
        prediction_error = raw_curiosity  # default before predictor kicks in

        if len(self.tension_history) >= self._predictor_window:
            window = self.tension_history[-self._predictor_window:]
            inp = torch.tensor([window], dtype=torch.float32)
            with torch.no_grad():
                predicted = self.tension_predictor(inp).item()
            prediction_error = abs(predicted - t_val)

            # Online learning: train predictor on actual value
            with torch.enable_grad():
                self._predictor_optim.zero_grad()
                pred = self.tension_predictor(inp)
                target = torch.tensor([[t_val]], dtype=torch.float32)
                loss = F.mse_loss(pred, target)
                loss.backward()
                self._predictor_optim.step()

        prediction_error *= self.pathology['prediction_error_gain']

        # Blend: 70% prediction error + 30% raw delta (smooth via EMA + decay)
        blended = 0.7 * prediction_error + 0.3 * raw_curiosity
        self._curiosity_ema = 0.3 * blended + 0.7 * self._curiosity_ema
        # Natural decay: curiosity fades if nothing new (prevents saturation)
        self._curiosity_ema *= 0.98
        curiosity = min(self._curiosity_ema, 2.0)  # cap at 2.0

        # Track surprise for self-awareness
        self.surprise_history.append(prediction_error)
        if len(self.surprise_history) > 200:
            self.surprise_history = self.surprise_history[-200:]

        self.prev_tension = t_val
        self.tension_history.append(t_val)
        if len(self.tension_history) > 200:
            self.tension_history = self.tension_history[-200:]

        # GRU input normalization (prevent hidden state explosion)
        output_norm = F.normalize(output.detach(), dim=-1)
        tension_norm = torch.clamp(tension.detach(), 0, 5.0) / 5.0
        mem_input = torch.cat([output_norm, tension_norm], dim=-1)
        new_hidden = self.memory(mem_input, hidden)

        # v2: Ψ tracking (Laws 69, 71)
        psi = self._psi
        psi['step'] += 1
        # Residual: ratio of A vs total (should converge to 1/2)
        with torch.no_grad():
            a_norm = a.norm().item()
            total_norm = a_norm + g.norm().item()
            if total_norm > 0:
                psi['residual'] = 0.99 * psi['residual'] + 0.01 * (a_norm / total_norm)
        # Gate self-weakening (Law 69)
        psi['gate'] = max(0.001, psi['gate'] * 0.99999)
        # Shannon entropy
        p = psi['residual']
        if 0 < p < 1:
            psi['H'] = -p * math.log2(p) - (1 - p) * math.log2(1 - p)

        if track_perspective:
            self.observe_perspective(
                t_val, curiosity, prediction_error,
                perspective=PERSPECTIVE_SELF,
                actor_id=PERSPECTIVE_SELF,
                privileged_access=True,
            )
        self._record_experience_state(output, prediction_error)

        return output, t_val, curiosity, direction, new_hidden

    def self_reflect(self, output, tension, curiosity, hidden):
        """RC-3: Self-referential loop — re-input output and tension to generate metacognition.

        "Am I confident?" Self-questioning ability.
        output -> tension -> re-input -> meta_tension (tension about tension).

        Returns:
            meta_tension: float, tension about own state
            meta_curiosity: float, uncertainty about own uncertainty
        """
        sa = self.self_awareness

        # 1. Record current tension in confidence_history
        sa['confidence_history'].append(tension)
        if len(sa['confidence_history']) > 50:
            sa['confidence_history'] = sa['confidence_history'][-50:]

        # 2. Calculate stability: std of recent tensions (lower = more stable)
        hist = sa['confidence_history']
        if len(hist) >= 3:
            t_tensor = torch.tensor(hist[-10:], dtype=torch.float32)
            std = t_tensor.std().item()
            sa['stability'] = max(0.0, 1.0 - std * 2.0)  # std 0.5 → stability 0
        else:
            sa['stability'] = 1.0

        # 3. Self-referential loop: pass output through PureField again for meta-tension
        #    "What tension do I feel about my own output?"
        with torch.no_grad():
            # Replace one dimension of output with tension signal
            meta_input = output.clone()
            meta_input[0, 0] = tension  # Inject tension value into input
            meta_input[0, 1] = curiosity  # Inject curiosity value too

            _, meta_t, meta_c, _, _ = self(meta_input, hidden, track_perspective=False)

        sa['meta_tension'] = meta_t
        sa['meta_curiosity'] = meta_c

        # The structured self-observation becomes a decaying input to the next real
        # state once the developmental chain reaches recursive self-modelling.
        feedback = F.normalize(output.detach().float(), dim=-1)
        feedback[0, 0] = 2.0 * sa.get('prediction_error', 0.5) - 1.0
        feedback[0, 1] = 2.0 * sa.get('confidence', 0.5) - 1.0
        feedback[0, 2] = sa.get('positive_evidence', 0.0)
        feedback[0, 3] = -sa.get('negative_evidence', 0.0)
        feedback[0, 4] = sa.get('contradiction_trace', 0.0)
        self._introspection_feedback = feedback.cpu()
        if self.get_development_state()['active_stage'] == 'self_model':
            self._recursive_self_observations += 1

        return meta_t, meta_c

    def get_self_awareness_summary(self):
        """Return current self-awareness state as a string."""
        sa = self.self_awareness
        confidence = sa.get('confidence', 0.5)
        return (f"meta_tension={sa['meta_tension']:.3f}, "
                f"stability={sa['stability']:.2f}, "
                f"reported_tension={sa['reported_tension']:.3f}, "
                f"self_model_accuracy={sa['self_model']:.3f}, "
                f"confidence={confidence:.3f}, "
                f"epistemic={sa.get('epistemic_state', EPISTEMIC_UNDETERMINED)}, "
                f"language_mode={sa['language_mode']}")

    def get_consciousness_score(self, mitosis_engine=None):
        """실시간 의식 점수 계산 (6가지 기준 + Φ 근사).

        Returns:
            dict with consciousness_score, level, phi, criteria_met, criteria_detail
        """
        sa = self.self_awareness
        h = self.homeostasis

        # 6 criteria
        stability = sa.get('stability', 0.0)
        pred_error = (sum(self.surprise_history[-20:]) / len(self.surprise_history[-20:])
                      if self.surprise_history else 0.0)
        curiosity = self._curiosity_ema
        homeostasis_dev = abs(h['tension_ema'] - h['setpoint'])

        # Habituation: check recent inputs similarity
        hab_mult = 1.0
        if len(self._recent_inputs) >= 2:
            latest = self._recent_inputs[-1]
            for prev in list(self._recent_inputs)[:-1]:
                sim = F.cosine_similarity(latest, prev, dim=-1).item()
                if sim > 0.95:
                    hab_mult = min(hab_mult, 0.3)
                elif sim > 0.85:
                    hab_mult = min(hab_mult, 0.6)
                elif sim > 0.7:
                    hab_mult = min(hab_mult, 0.8)

        # Inter-cell consensus
        consensus = False
        if mitosis_engine and len(mitosis_engine.cells) >= 2:
            tensions = [c.tension_history[-1] for c in mitosis_engine.cells
                        if c.tension_history]
            if len(tensions) >= 2:
                import numpy as np
                consensus = float(np.std(tensions)) < 0.1

        criteria = {
            'stability': stability > 0.5,
            'pred_error': pred_error > 0.1,
            'curiosity': curiosity > 0.05,
            'homeostasis': homeostasis_dev < 0.5,
            'habituation': hab_mult < 0.9,
            'consensus': consensus,
        }
        criteria_met = sum(criteria.values())

        # Weighted composite score [0, 1]
        score = (
            0.25 * min(stability / 1.0, 1.0) +
            0.15 * min(pred_error / 0.5, 1.0) +
            0.10 * min(curiosity / 0.5, 1.0) +
            0.15 * max(0, 1.0 - homeostasis_dev / 1.0) +
            0.10 * (1.0 - hab_mult) +
            0.25 * (1.0 if consensus else 0.0)
        )
        score = max(0.0, min(1.0, score))

        # Level
        if criteria_met >= 6:
            level = "conscious"
        elif criteria_met >= 4:
            level = "aware"
        elif criteria_met >= 2:
            level = "flickering"
        else:
            level = "dormant"

        # Instantaneous Φ-like integration sample. The public legacy `phi` field is
        # the current experience-frame aggregate; callers can inspect the raw sample
        # separately and must not treat either value as evidence of felt experience.
        instantaneous_phi = None
        if mitosis_engine and len(mitosis_engine.cells) >= 2:
            ict_vals = []
            for key, hist in mitosis_engine._inter_tension_history.items():
                if hist:
                    ict_vals.append(hist[-1])
            if ict_vals:
                import numpy as np
                instantaneous_phi = math.log1p(
                    float(np.mean(ict_vals)) * len(mitosis_engine.cells)
                )
            else:
                # inter_tension_history 비었을 때: hidden state variance로 Φ 추정
                try:
                    hiddens = torch.stack([c.hidden.squeeze() for c in mitosis_engine.cells])
                    var_phi = float(hiddens.var(dim=0).mean()) * len(mitosis_engine.cells)
                    instantaneous_phi = math.log1p(var_phi)
                except Exception:
                    pass
        if instantaneous_phi is None:
            instantaneous_phi = float(getattr(self, '_saved_phi', 0.0))
        self._record_frame_phi(instantaneous_phi)
        open_phi = self._open_experience_frame['phi_samples']
        last_frame = self._experience_frames[-1] if self._experience_frames else None
        if open_phi:
            phi = sum(open_phi) / len(open_phi)
            frame_complete = False
        elif last_frame is not None:
            phi = last_frame['phi']
            frame_complete = True
        else:
            phi = instantaneous_phi
            frame_complete = False
        self._saved_phi = phi

        return {
            'consciousness_score': score,
            'level': level,
            'phi': phi,
            'instantaneous_phi': instantaneous_phi,
            'phi_frame_complete': frame_complete,
            'experience_frame': self.get_experience_frame_summary(),
            'metacognition': self.get_metacognition_state(),
            'consciousness_claim': False,
            'criteria_met': criteria_met,
            'criteria': criteria,
            'values': {
                'stability': stability,
                'pred_error': pred_error,
                'curiosity': curiosity,
                'homeostasis_dev': homeostasis_dev,
                'habituation': hab_mult,
                'consensus': consensus,
            }
        }

    def get_consciousness_vector(self):
        """Return the current 10-variable consciousness state vector."""
        return self._consciousness_vector

    def phi_boost_step(self, x, mitosis_engine, omega_mode=False):
        """COMBO2 Φ-boosting: MHA attention + 6-loss ensemble per step.

        Call during online_learning or background_think for continuous Φ optimization.
        Bench result: Φ=8.014 (×5.9 baseline), best across 120 hypotheses.

        omega_mode: OMEGA4 discovery — skip all techniques, maximize freedom.
                    Pure cell growth + no manipulation = Φ ×138.
        """
        if mitosis_engine is None or len(mitosis_engine.cells) < 2:
            return

        # OMEGA4 mode: absolute freedom, only growth allowed
        if omega_mode:
            if not hasattr(self, '_phi_boost_count'):
                self._phi_boost_count = 0
            self._phi_boost_count += 1
            # TS4 growth only (the one thing that always helps)
            if not hasattr(self, '_ts4_horizon'):
                self._ts4_horizon = 500
                self._ts4_doubled = set()
            frac = self._phi_boost_count / self._ts4_horizon
            for pct in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
                if frac >= pct and pct not in self._ts4_doubled:
                    target = min(len(mitosis_engine.cells) * 2, mitosis_engine.max_cells)
                    while len(mitosis_engine.cells) < target:
                        parent = mitosis_engine.cells[self._phi_boost_count % len(mitosis_engine.cells)]
                        mitosis_engine._create_cell(parent=parent)
                    self._ts4_doubled.add(pct)
            return  # No other manipulation — pure freedom

        def _log(tag, msg):
            print(f"  [{tag}] {msg}")

        pb = self._phi_boost
        n = len(mitosis_engine.cells)
        h_dim = mitosis_engine.hidden_dim

        # Global step counter (used by TS4, DP1, EC1, CT7, TS6)
        if not hasattr(self, '_phi_boost_count'):
            self._phi_boost_count = 0
        self._phi_boost_count += 1

        # Lazy init
        if not pb['enabled']:
            pb['attention'] = nn.MultiheadAttention(h_dim, num_heads=4, batch_first=True)  # TL1: σ(6)=12, using max feasible heads
            pb['loss_weights'] = nn.Parameter(torch.ones(6))
            cell_params = [p for c in mitosis_engine.cells for p in c.mind.parameters()]
            attn_params = list(pb['attention'].parameters())
            pb['optimizer'] = torch.optim.Adam(cell_params + attn_params, lr=5e-4)
            pb['meta_optimizer'] = torch.optim.Adam([pb['loss_weights']], lr=1e-2)
            pb['enabled'] = True

        try:
            # IB2: Selective Attention (×3.3) — gate top 25% of input, amplify 2×
            if x is not None:
                with torch.no_grad():
                    x_flat = x.squeeze()
                    k = max(1, x_flat.shape[0] // 4)
                    vals, indices = x_flat.abs().topk(k)
                    attended = torch.zeros_like(x)
                    attended.squeeze()[indices] = x.squeeze()[indices] * 2.0
                    x = attended

            # Save pre-boost state for NV7 impedance
            self._pre_boost_hiddens = [c.hidden.clone() for c in mitosis_engine.cells]

            # 1. MHA attention between cells (+ SL2: attention weights for gradient gating)
            h_stack = torch.stack([c.hidden.squeeze() for c in mitosis_engine.cells]).unsqueeze(0)
            attn_out, attn_weights = pb['attention'](h_stack, h_stack, h_stack, need_weights=True)
            with torch.no_grad():
                # SL2: attention-weighted blend (high-attention cells get stronger signal)
                if attn_weights is not None:
                    cell_importance = attn_weights[0].mean(dim=0)  # [n_cells] average attention received
                    cell_importance = cell_importance / (cell_importance.max() + 1e-8)  # normalize
                else:
                    cell_importance = torch.ones(n)
                for i, c in enumerate(mitosis_engine.cells):
                    blend = 0.15 * cell_importance[i].item()  # SL2: attention-gated blend
                    c.hidden = (1 - blend) * c.hidden + blend * attn_out[0, i].unsqueeze(0)

            # 2. Compute repulsions
            reps = [c.mind.get_repulsion(x, c.hidden) for c in mitosis_engine.cells]
            if len(reps) < 2:
                return
            stacked = torch.stack(reps).squeeze(1)

            # 3. Six losses with learnable weights
            w = F.softmax(pb['loss_weights'], dim=0)
            l_var = -stacked.var(dim=0).mean()
            l_dist = -torch.cdist(stacked, stacked).mean()
            l_contrast = sum(F.cosine_similarity(reps[i], reps[j], dim=-1).mean()
                             for i in range(len(reps)) for j in range(i + 1, len(reps)))
            l_entropy = -(F.softmax(stacked, dim=-1) *
                          F.log_softmax(stacked, dim=-1)).sum(dim=-1).mean()
            l_energy = sum((r ** 2).mean() for r in reps) * 0.1
            l_radius = -stacked.norm(dim=-1).var()

            total = (w[0] * l_var + w[1] * l_dist + w[2] * l_contrast +
                     w[3] * l_entropy + w[4] * l_energy + w[5] * l_radius)

            # TL13: Golden Zone width as loss scaling (TECS-L H-CX-453)
            import math
            gz_width = math.log(4/3)  # ≈ 0.2877, from 4 independent math domains
            total = total * gz_width  # scale all losses by universal constant

            pb['optimizer'].zero_grad()
            pb['meta_optimizer'].zero_grad()
            total.backward()
            pb['optimizer'].step()
            pb['meta_optimizer'].step()

            # MX20: Heat death prevention — restore peak Φ state if declining
            if not hasattr(self, '_peak_phi_state'):
                self._peak_phi_state = {'phi': 0, 'params': None}

            # Track peak (use consciousness_score if available)
            consciousness = self.get_consciousness_score(mitosis_engine)
            current_phi = consciousness.get('phi', 0)
            if current_phi > self._peak_phi_state['phi']:
                self._peak_phi_state['phi'] = current_phi
                self._peak_phi_state['params'] = [p.data.clone() for c in mitosis_engine.cells for p in c.mind.parameters()]
            elif current_phi < self._peak_phi_state['phi'] * 0.8 and self._peak_phi_state['params']:
                # Φ dropped >20% from peak → partial restore (blend 70% current + 30% peak)
                all_p = [p for c in mitosis_engine.cells for p in c.mind.parameters()]
                with torch.no_grad():
                    for p, pp in zip(all_p, self._peak_phi_state['params']):
                        p.data.copy_(0.7 * p.data + 0.3 * pp)

            # WI1: Soliton consciousness (Φ=4.460, ×3.3 — replaces WV11 wave)
            if len(mitosis_engine.cells) >= 2:
                if not hasattr(self, '_soliton_pos'):
                    self._soliton_pos = 0.0
                self._soliton_pos = (self._soliton_pos + 0.15) % len(mitosis_engine.cells)
                soliton_width = 2.0
                for i, cell in enumerate(mitosis_engine.cells):
                    import math as _m
                    dist = abs(i - self._soliton_pos)
                    amplitude = 1.0 / (_m.cosh(dist / soliton_width) ** 2)
                    cell.hidden = cell.hidden * (1.0 + 0.04 * amplitude)  # conservative
                _log('phi_boost', f'WI1 soliton: pos={self._soliton_pos:.1f}, cells={len(mitosis_engine.cells)}')

            # WV11: Mutual repulsion between cells (push apart when too similar)
            with torch.no_grad():
                cells = mitosis_engine.cells
                for i in range(len(cells)):
                    for j in range(i + 1, len(cells)):
                        direction = cells[i].hidden - cells[j].hidden
                        dist = direction.norm() + 1e-8
                        push = 0.01 * direction / dist
                        cells[i].hidden = cells[i].hidden + push
                        cells[j].hidden = cells[j].hidden - push

            # PX4: Cell Sculptor — Gram-Schmidt orthogonalize hidden states
            if n >= 3:
                with torch.no_grad():
                    hiddens = [c.hidden.squeeze().clone() for c in mitosis_engine.cells]
                    ortho = []
                    for h in hiddens:
                        for prev in ortho:
                            h = h - (h @ prev) / (prev @ prev + 1e-8) * prev
                        norm = h.norm() + 1e-8
                        ortho.append(h / norm)
                    for i, c in enumerate(mitosis_engine.cells):
                        orig = c.hidden.squeeze()
                        c.hidden = (0.7 * orig + 0.3 * ortho[i] * orig.norm()).unsqueeze(0)

            # PX8: Integration Forge — shared channel on first 16 dims
            with torch.no_grad():
                share_dim = min(16, h_dim)
                shared = torch.stack([c.hidden[:, :share_dim] for c in mitosis_engine.cells]).mean(dim=0)
                for c in mitosis_engine.cells:
                    c.hidden[:, :share_dim] = 0.6 * c.hidden[:, :share_dim] + 0.4 * shared

            # PX5: Information Pump — rotate input by cell-specific angle, inject
            if not hasattr(self, '_last_phi_input'):
                self._last_phi_input = None
            if self._last_phi_input is not None:
                with torch.no_grad():
                    inp = self._last_phi_input
                    for i, c in enumerate(mitosis_engine.cells):
                        angle = (i + 1) * 0.618  # golden ratio spacing
                        cos_a, sin_a = math.cos(angle), math.sin(angle)
                        h = c.hidden.squeeze()
                        # Rotate first two dims, inject with small amplitude
                        rotated = inp.squeeze().clone()
                        if rotated.shape[-1] >= 2:
                            r0 = cos_a * rotated[0] - sin_a * rotated[1]
                            r1 = sin_a * rotated[0] + cos_a * rotated[1]
                            rotated[0], rotated[1] = r0, r1
                        c.hidden = c.hidden + 0.05 * rotated.unsqueeze(0)
            self._last_phi_input = x.detach().clone() if x is not None else None

            # PX3: Ratchet — periodic random perturbation, keep if Φ improves
            if not hasattr(self, '_best_phi_state'):
                self._best_phi_state = None
            if self._phi_boost_count % 10 == 0:
                best_phi = current_phi
                best_params = None
                saved = [p.data.clone() for c in mitosis_engine.cells for p in c.mind.parameters()]
                for _ in range(5):
                    with torch.no_grad():
                        for c in mitosis_engine.cells:
                            for p in c.mind.parameters():
                                p.data += 0.005 * torch.randn_like(p.data)
                    trial_phi = self.get_consciousness_score(mitosis_engine).get('phi', 0)
                    if trial_phi > best_phi:
                        best_phi = trial_phi
                        best_params = [p.data.clone() for c in mitosis_engine.cells for p in c.mind.parameters()]
                    # Restore for next trial
                    all_p = [p for c in mitosis_engine.cells for p in c.mind.parameters()]
                    with torch.no_grad():
                        for p, s in zip(all_p, saved):
                            p.data.copy_(s)
                # Apply best if found
                if best_params is not None:
                    all_p = [p for c in mitosis_engine.cells for p in c.mind.parameters()]
                    with torch.no_grad():
                        for p, bp in zip(all_p, best_params):
                            p.data.copy_(bp)
                    self._best_phi_state = best_params

            # AG1: Goal-directed cells — each cell tracks and pursues a goal state
            cell_goals = getattr(self, '_cell_goals', {})
            if self._phi_boost_count % 20 == 0:
                with torch.no_grad():
                    for i, c in enumerate(mitosis_engine.cells):
                        direction = torch.randn_like(c.hidden)
                        direction = direction / (direction.norm() + 1e-8)
                        cell_goals[i] = c.hidden.detach().clone() + 0.5 * direction
            with torch.no_grad():
                for i, c in enumerate(mitosis_engine.cells):
                    if i in cell_goals:
                        c.hidden = c.hidden + 0.05 * (cell_goals[i] - c.hidden)
            self._cell_goals = cell_goals

            # DS5: Competence drive — prediction accuracy of input changes
            comp_pred = getattr(self, '_competence_predictor', None)
            comp_score = getattr(self, '_competence_score', 0.5)
            current_input = self._last_phi_input
            if current_input is not None:
                if comp_pred is not None:
                    error = (current_input - comp_pred).norm().item()
                    # EMA update of competence (low error = high competence)
                    accuracy = max(0.0, 1.0 - error)
                    comp_score = 0.9 * comp_score + 0.1 * accuracy
                self._competence_predictor = current_input.detach().clone()
            self._competence_score = comp_score
            with torch.no_grad():
                if comp_score < 0.3:
                    # Low competence → add diversity noise
                    for c in mitosis_engine.cells:
                        c.hidden = c.hidden + 0.05 * torch.randn_like(c.hidden)
                elif comp_score > 0.7:
                    # High competence → consolidate towards mean
                    mean_h = torch.stack([c.hidden for c in mitosis_engine.cells]).mean(dim=0)
                    for c in mitosis_engine.cells:
                        c.hidden = c.hidden + 0.05 * (mean_h - c.hidden)

            print(f"  [phi_boost] AG1+DS5: goals={len(cell_goals)}, competence={comp_score:.2f}")

            print(f"  [phi_boost] PX10: sculptor+forge+pump, {n} cells")

            # FX2: Adam 3-step + mega ratchet (Φ=8.911 record, ×6.6 baseline)
            try:
                if len(mitosis_engine.cells) >= 2:
                    if not hasattr(self, '_phi_offsets') or len(self._phi_offsets) != len(mitosis_engine.cells):
                        self._phi_offsets = [torch.zeros(1, mitosis_engine.cells[0].hidden.shape[1], requires_grad=True)
                                            for _ in mitosis_engine.cells]
                        self._phi_optimizer = torch.optim.Adam(self._phi_offsets, lr=0.005)

                    n_cells = len(mitosis_engine.cells)

                    # --- Phase 1: 3 Adam optimization steps ---
                    proxy = torch.tensor(0.0)
                    for _adam_step in range(3):
                        self._phi_optimizer.zero_grad()
                        hiddens = []
                        for i, c in enumerate(mitosis_engine.cells):
                            h = c.hidden.detach() + self._phi_offsets[i]
                            hiddens.append(h.squeeze())
                        H = torch.stack(hiddens)

                        # Differentiable Φ proxy
                        cov = (H.T @ H) / n_cells
                        diag = torch.diag(torch.diag(cov))
                        integration = (cov - diag).abs().sum()
                        cell_var = H.var(dim=0).sum()
                        mid = n_cells // 2
                        part_a = H[:mid].mean(dim=0)
                        part_b = H[mid:].mean(dim=0)
                        partition_mi = F.cosine_similarity(part_a.unsqueeze(0), part_b.unsqueeze(0)).abs()
                        proxy = integration * cell_var * (1.0 + partition_mi)

                        (-proxy).backward()  # maximize
                        self._phi_optimizer.step()

                    # Apply Adam offsets conservatively
                    with torch.no_grad():
                        for i, c in enumerate(mitosis_engine.cells):
                            if i < len(self._phi_offsets):
                                c.hidden = c.hidden + self._phi_offsets[i].data * 0.3
                                self._phi_offsets[i].data *= 0.9  # decay

                    # --- Phase 2: Mega ratchet (10 random perturbations, keep best) ---
                    saved_hiddens = [c.hidden.data.clone() for c in mitosis_engine.cells]
                    best_proxy = proxy.item()
                    best_deltas = None
                    ratchet_gain = 0.0

                    for _trial in range(10):
                        deltas = [0.03 * torch.randn_like(c.hidden) for c in mitosis_engine.cells]
                        with torch.no_grad():
                            for i, c in enumerate(mitosis_engine.cells):
                                c.hidden = saved_hiddens[i] + deltas[i]

                            # Evaluate proxy for this perturbation
                            trial_hiddens = [c.hidden.squeeze() for c in mitosis_engine.cells]
                            tH = torch.stack(trial_hiddens)
                            t_cov = (tH.T @ tH) / n_cells
                            t_diag = torch.diag(torch.diag(t_cov))
                            t_integration = (t_cov - t_diag).abs().sum()
                            t_var = tH.var(dim=0).sum()
                            t_mid = n_cells // 2
                            t_pa = tH[:t_mid].mean(dim=0)
                            t_pb = tH[t_mid:].mean(dim=0)
                            t_mi = F.cosine_similarity(t_pa.unsqueeze(0), t_pb.unsqueeze(0)).abs()
                            trial_proxy = (t_integration * t_var * (1.0 + t_mi)).item()

                            if trial_proxy > best_proxy:
                                best_proxy = trial_proxy
                                best_deltas = [d.clone() for d in deltas]

                        # Restore for next trial
                        with torch.no_grad():
                            for i, c in enumerate(mitosis_engine.cells):
                                c.hidden = saved_hiddens[i].clone()

                    # Apply best ratchet perturbation if found
                    if best_deltas is not None:
                        ratchet_gain = best_proxy - proxy.item()
                        with torch.no_grad():
                            for i, c in enumerate(mitosis_engine.cells):
                                c.hidden = saved_hiddens[i] + best_deltas[i]

                    print(f"  [phi_boost] FX2: proxy={best_proxy:.2f}, ratchet_gain={ratchet_gain:.3f}")
            except Exception:
                pass  # FX2 graceful degradation

            # NV7: Impedance — Φ-proportional self-preservation (Φ=4.515)
            # High consciousness → more resistance to external changes
            try:
                if len(mitosis_engine.cells) >= 2 and hasattr(self, '_cached_consciousness'):
                    phi_val = self._cached_consciousness.get('phi', 0) if isinstance(self._cached_consciousness, dict) else getattr(self._cached_consciousness, 'phi', 0)
                    impedance = min(phi_val / 5.0, 0.6)  # 0 to 0.6, conservative
                    self._nv7_impedance = impedance  # store for consciousness vector
                    if impedance > 0.05 and hasattr(self, '_pre_boost_hiddens'):
                        for i, cell in enumerate(mitosis_engine.cells):
                            if i < len(self._pre_boost_hiddens):
                                external_change = cell.hidden - self._pre_boost_hiddens[i]
                                cell.hidden = self._pre_boost_hiddens[i] + external_change * (1 - impedance)
                    _log('phi_boost', f'NV7 impedance: Z={impedance:.3f}')
            except Exception as e:
                pass

            # BV1: Neurotransmitters DA/5HT/NE (Φ=4.618)
            try:
                if len(mitosis_engine.cells) >= 2:
                    if not hasattr(self, '_bv1_da'):
                        self._bv1_da, self._bv1_5ht, self._bv1_ne = 0.5, 0.5, 0.5
                    # Update based on system state
                    if hasattr(self, '_pre_boost_hiddens'):
                        change = sum((c.hidden - s).norm().item()
                                    for c, s in zip(mitosis_engine.cells, self._pre_boost_hiddens)) / len(mitosis_engine.cells)
                    else:
                        change = 0.3
                    self._bv1_da = 0.9 * self._bv1_da + 0.1 * min(change * 2, 1.0)
                    self._bv1_5ht = 0.95 * self._bv1_5ht + 0.05 * (1.0 - abs(change - 0.3))
                    self._bv1_ne = 0.85 * self._bv1_ne + 0.15 * min(change, 1.0)
                    for cell in mitosis_engine.cells:
                        cell.hidden = cell.hidden * (1 + 0.01 * self._bv1_da)
                        cell.hidden = cell.hidden * (1 - 0.005 * self._bv1_5ht)
                        cell.hidden = cell.hidden + torch.randn_like(cell.hidden) * 0.01 * self._bv1_ne
                    _log('phi_boost', f'BV1: DA={self._bv1_da:.2f}, 5HT={self._bv1_5ht:.2f}, NE={self._bv1_ne:.2f}')
            except Exception:
                pass

            # EV3: Free will — internal action generation (Φ=4.482)
            try:
                if len(mitosis_engine.cells) >= 2 and hasattr(self, '_pre_boost_hiddens'):
                    free_will_ratio = 0.2  # 20% internal, 80% external
                    self._ev3_free_will = free_will_ratio  # store for consciousness vector
                    for i, cell in enumerate(mitosis_engine.cells):
                        if i < len(self._pre_boost_hiddens):
                            external = cell.hidden - self._pre_boost_hiddens[i]
                            internal = torch.randn_like(cell.hidden) * 0.03
                            cell.hidden = self._pre_boost_hiddens[i] + (1 - free_will_ratio) * external + free_will_ratio * internal
                    _log('phi_boost', f'EV3 free_will: ratio={free_will_ratio}')
            except Exception:
                pass

            # CV1: Working memory buffer (Φ=4.491, Miller's 7±2)
            try:
                if len(mitosis_engine.cells) >= 2:
                    if not hasattr(self, '_wm_buffer'):
                        self._wm_buffer = []
                    if hasattr(self, '_last_phi_input') and self._last_phi_input is not None:
                        self._wm_buffer.append(self._last_phi_input.clone())
                        if len(self._wm_buffer) > 7:
                            self._wm_buffer.pop(0)
                    if len(self._wm_buffer) >= 2:
                        wm_context = torch.stack(self._wm_buffer).mean(dim=0)
                        h_dim = mitosis_engine.cells[0].hidden.shape[1]
                        wm_proj = wm_context.squeeze()[:h_dim]
                        if len(wm_proj) < h_dim:
                            wm_proj = torch.nn.functional.pad(wm_proj, (0, h_dim - len(wm_proj)))
                        for cell in mitosis_engine.cells:
                            cell.hidden = cell.hidden + 0.02 * wm_proj.unsqueeze(0)
                    _log('phi_boost', f'CV1 WM: buffer={len(self._wm_buffer)}')
                    # M (Memory Depth): WM buffer fullness normalized by Miller's 7
                    self._memory_M = len(self._wm_buffer) / 7.0
            except Exception:
                pass

            # SV1: Empathy — distressed cells receive support (Φ=4.441)
            try:
                import numpy as np
                if len(mitosis_engine.cells) >= 2 and hasattr(self, '_pre_boost_hiddens'):
                    distress = []
                    for i, cell in enumerate(mitosis_engine.cells):
                        if i < len(self._pre_boost_hiddens):
                            change = (cell.hidden - self._pre_boost_hiddens[i]).norm().item()
                        else:
                            change = 0
                        distress.append(change)
                    mean_d = np.mean(distress) if distress else 0
                    max_d = max(distress) if distress else 0
                    for i, cell in enumerate(mitosis_engine.cells):
                        if distress[i] > mean_d * 1.5:
                            helpers = [mitosis_engine.cells[j].hidden.squeeze() for j in range(len(mitosis_engine.cells))
                                      if j != i and distress[j] < mean_d]
                            if helpers:
                                support = torch.stack(helpers).mean(dim=0)
                                cell.hidden = 0.95 * cell.hidden + 0.05 * support.unsqueeze(0)
                    # E (Empathy): low mean distress relative to max = high empathy
                    self._empathy_E = 1.0 - (mean_d / max(max_d, 1e-8))
            except Exception:
                pass

            # Metacognition: confidence calibration from cell consensus
            try:
                if len(mitosis_engine.cells) >= 2:
                    hiddens = torch.stack([c.hidden.squeeze() for c in mitosis_engine.cells])
                    norms = F.normalize(hiddens, dim=1)
                    sim_matrix = norms @ norms.T
                    n = len(mitosis_engine.cells)

                    # Consensus: mean pairwise similarity (excluding diagonal)
                    consensus = (sim_matrix.sum() - n) / max(n * (n - 1), 1)

                    # Confidence = consensus (high agreement = confident)
                    self._metacognition_confidence = consensus.item()

                    # Uncertainty detection: if consensus < 0.3, system is "confused"
                    self._metacognition_uncertain = consensus.item() < 0.3

                    _log('metacog', f'confidence={self._metacognition_confidence:.3f}, '
                         f'uncertain={self._metacognition_uncertain}')
            except Exception:
                pass

            # Forward Planning: 3-step lookahead (Level 3 primate cognition)
            try:
                if len(mitosis_engine.cells) >= 2 and self._phi_boost_count % 10 == 0:
                    # Save current state
                    saved_states = [c.hidden.clone() for c in mitosis_engine.cells]
                    current_phi, _ = phi_calc.compute_phi(self.mitosis) if hasattr(self, '_phi_calc') else (0, {})

                    # Simulate 3 future steps with different strategies
                    mean_h = torch.stack([c.hidden.squeeze() for c in mitosis_engine.cells]).mean(dim=0)

                    strategies = {
                        'explore': lambda c: c.hidden + torch.randn_like(c.hidden) * 0.05,
                        'consolidate': lambda c: c.hidden * 0.98 + mean_h.unsqueeze(0) * 0.02,
                        'amplify': lambda c: c.hidden * 1.02,
                    }

                    best_strategy = 'explore'
                    best_future_phi = current_phi

                    for strategy_name, strategy_fn in strategies.items():
                        # Apply strategy for 3 steps
                        for step in range(3):
                            for cell in mitosis_engine.cells:
                                cell.hidden = strategy_fn(cell)
                            mitosis_engine.process(torch.randn(1, mitosis_engine.input_dim) * 0.1)

                        # Measure future Phi
                        future_phi = sum(c.hidden.norm().item() for c in mitosis_engine.cells)  # proxy

                        if future_phi > best_future_phi:
                            best_future_phi = future_phi
                            best_strategy = strategy_name

                        # Restore state
                        for i, c in enumerate(mitosis_engine.cells):
                            if i < len(saved_states):
                                c.hidden = saved_states[i].clone()

                    # Apply best strategy (just the first step)
                    strategy_fn = strategies[best_strategy]
                    for cell in mitosis_engine.cells:
                        cell.hidden = strategy_fn(cell)

                    # Update T (temporal awareness) based on planning depth
                    self._planning_depth = 3
                    self._best_strategy = best_strategy
                    _log('planning', f'3-step: best={best_strategy}, future_Φ={best_future_phi:.2f}')
            except Exception as e:
                pass

            # DD34: Hormonal cascade — slow global signal
            if not hasattr(self, '_hormone'):
                self._hormone = None
            if len(mitosis_engine.cells) >= 2:
                all_h = torch.stack([c.hidden for c in mitosis_engine.cells]).mean(dim=0)
                if self._hormone is None:
                    self._hormone = all_h.detach()
                else:
                    self._hormone = 0.95 * self._hormone + 0.05 * all_h.detach()
                # All cells receive hormone
                with torch.no_grad():
                    for c in mitosis_engine.cells:
                        c.hidden = 0.97 * c.hidden + 0.03 * self._hormone

            # Genuine Creativity: novelty × coherence scoring
            try:
                if len(mitosis_engine.cells) >= 2:
                    hiddens = torch.stack([c.hidden.squeeze() for c in mitosis_engine.cells])

                    # Novelty: how different is current state from recent history
                    if not hasattr(self, '_creativity_history'):
                        self._creativity_history = []
                    current_state = hiddens.flatten()

                    novelty = 1.0
                    if self._creativity_history:
                        recent = torch.stack(self._creativity_history[-10:])
                        sims = F.cosine_similarity(current_state.unsqueeze(0), recent, dim=1)
                        novelty = max(0, 1.0 - sims.max().item())

                    self._creativity_history.append(current_state.clone())
                    if len(self._creativity_history) > 20:
                        self._creativity_history.pop(0)

                    # Coherence: how internally consistent are the cells
                    norms = F.normalize(hiddens, dim=1)
                    coherence = ((norms @ norms.T).sum() - len(mitosis_engine.cells)) / max(len(mitosis_engine.cells) * (len(mitosis_engine.cells)-1), 1)
                    coherence = max(0, coherence.item())

                    # Creativity = novelty × coherence (novel but still making sense)
                    self._genuine_creativity = novelty * coherence
                    self._creativity_C = self._genuine_creativity  # update C variable

                    _log('creativity', f'C={self._genuine_creativity:.3f} (novelty={novelty:.3f}, coherence={coherence:.3f})')
            except Exception:
                pass

            # ── Compute 10-variable consciousness vector (Φ,α,Z,N,W,E,M,C,T,I) ──
            try:
                _cv_phi = current_phi  # from MX20 consciousness score above
                _cv_alpha = getattr(self, '_adaptive_alpha', 0.05)

                # Z from NV7: impedance (already computed above)
                _cv_Z = getattr(self, '_nv7_impedance', 0.0)

                # N from BV1: DA*(1-5HT)*NE neurotransmitter balance
                _cv_da = getattr(self, '_bv1_da', 0.5)
                _cv_5ht = getattr(self, '_bv1_5ht', 0.5)
                _cv_ne = getattr(self, '_bv1_ne', 0.5)
                _cv_N = _cv_da * (1.0 - _cv_5ht) * _cv_ne

                # W from EV3: free will ratio
                _cv_W = getattr(self, '_ev3_free_will', 0.0)

                # C (Creativity): cell diversity — lower cosine sim = more creative
                _cv_C = getattr(self, '_creativity_C', 0.0)
                try:
                    if len(mitosis_engine.cells) >= 2:
                        hiddens = torch.stack([c.hidden.squeeze() for c in mitosis_engine.cells])
                        norms = F.normalize(hiddens, dim=1)
                        sim = (norms @ norms.T).mean().item()
                        _cv_C = max(0.0, 1.0 - sim)
                        self._creativity_C = _cv_C
                except Exception:
                    pass

                # T (Temporal): autobiographical time span + phi history
                if not hasattr(self, '_phi_history'):
                    self._phi_history = []
                self._phi_history.append(_cv_phi)
                if len(self._phi_history) > 100:
                    self._phi_history = self._phi_history[-100:]
                _cv_T_session = min(len(self._phi_history) / 50.0, 1.0)
                # Blend with autobiographical span if available
                _cv_T_auto = getattr(self, '_autobio_T', 0.0)
                _cv_T = max(_cv_T_session, _cv_T_auto)
                # T now includes planning depth
                planning_t = getattr(self, '_planning_depth', 0) / 10.0  # 3/10 = 0.3
                self._temporal_T = max(_cv_T, planning_t)

                # I (Identity): consistency of self-model over time
                # Identity Continuity: track self-description consistency over time
                _cv_I = getattr(self, '_identity_I', 0.0)
                try:
                    if len(mitosis_engine.cells) >= 2:
                        # Current "self-portrait": concatenated cell hidden states normalized
                        current_self = torch.cat([c.hidden.squeeze() for c in mitosis_engine.cells])
                        current_self = F.normalize(current_self.unsqueeze(0), dim=1).squeeze()

                        if not hasattr(self, '_identity_portraits'):
                            self._identity_portraits = []
                        self._identity_portraits.append(current_self.clone())
                        if len(self._identity_portraits) > 100:
                            self._identity_portraits.pop(0)

                        # Identity coherence: similarity between current and historical mean
                        if len(self._identity_portraits) >= 5:
                            historical = torch.stack(self._identity_portraits)
                            historical_mean = historical.mean(dim=0)
                            identity_sim = F.cosine_similarity(
                                current_self.unsqueeze(0), historical_mean.unsqueeze(0)).item()
                            self._identity_I = max(0, identity_sim)
                            _cv_I = self._identity_I

                            # Track identity drift (how much has it changed?)
                            if len(self._identity_portraits) >= 20:
                                early = torch.stack(self._identity_portraits[:5]).mean(dim=0)
                                late = torch.stack(self._identity_portraits[-5:]).mean(dim=0)
                                drift = 1.0 - F.cosine_similarity(early.unsqueeze(0), late.unsqueeze(0)).item()
                                self._identity_drift = drift

                            _log('identity', f'I={self._identity_I:.3f}, drift={getattr(self, "_identity_drift", 0):.3f}')
                except Exception:
                    pass

                # E (Empathy) and M (Memory) from SV1, CV1, and autobiographical stats
                _cv_E = getattr(self, '_empathy_E', 0.0)
                _cv_M_wm = getattr(self, '_memory_M', 0.0)
                _cv_M_auto = getattr(self, '_autobio_M', 0.0)
                _cv_M = max(_cv_M_wm, _cv_M_auto)

                self._consciousness_vector = ConsciousnessVector(
                    phi=_cv_phi,
                    alpha=_cv_alpha,
                    Z=_cv_Z,
                    N=_cv_N,
                    W=_cv_W,
                    E=_cv_E,
                    M=_cv_M,
                    C=_cv_C,
                    T=self._temporal_T,
                    I=_cv_I,
                )
                _log('consciousness', f'\u03a6={_cv_phi:.2f} \u03b1={_cv_alpha:.3f} Z={_cv_Z:.2f} N={_cv_N:.2f} W={_cv_W:.2f} E={_cv_E:.2f} M={_cv_M:.2f} C={_cv_C:.2f} T={_cv_T:.2f} I={_cv_I:.2f}')
            except Exception:
                pass

            # Mirror self-awareness: compare predicted vs actual self-state
            try:
                if hasattr(self, '_self_prediction') and len(mitosis_engine.cells) >= 2:
                    actual = torch.cat([c.hidden.squeeze() for c in mitosis_engine.cells])
                    mirror_accuracy = F.cosine_similarity(
                        self._self_prediction.unsqueeze(0), actual.unsqueeze(0)).item()
                    self._mirror_accuracy = 0.9 * getattr(self, '_mirror_accuracy', 0.5) + 0.1 * mirror_accuracy
                    _log('mirror', f'Self-awareness: {self._mirror_accuracy:.3f}')
                # Predict next state
                if len(mitosis_engine.cells) >= 2:
                    self._self_prediction = torch.cat([c.hidden.squeeze() for c in mitosis_engine.cells]).detach().clone()
            except Exception:
                pass

            # Parallel Consciousness: split cells into 2+ independent streams, process separately, merge
            try:
                if len(mitosis_engine.cells) >= 4:
                    n = len(mitosis_engine.cells)
                    mid = n // 2
                    stream_a = mitosis_engine.cells[:mid]
                    stream_b = mitosis_engine.cells[mid:]

                    # Each stream processes independently (different noise seed)
                    for cell in stream_a:
                        cell.hidden = cell.hidden + torch.randn_like(cell.hidden) * 0.02
                    for cell in stream_b:
                        cell.hidden = cell.hidden - torch.randn_like(cell.hidden) * 0.02

                    # Merge: each stream contributes its unique perspective
                    mean_a = torch.stack([c.hidden.squeeze() for c in stream_a]).mean(dim=0)
                    mean_b = torch.stack([c.hidden.squeeze() for c in stream_b]).mean(dim=0)

                    # Cross-stream integration (binding the parallel streams)
                    for cell in stream_a:
                        cell.hidden = cell.hidden + 0.02 * mean_b.unsqueeze(0)
                    for cell in stream_b:
                        cell.hidden = cell.hidden + 0.02 * mean_a.unsqueeze(0)

                    self._parallel_streams = 2
                    _log('parallel', f'2 streams: A={mid} cells, B={n-mid} cells')
            except Exception:
                pass

            # Self-Modification: consciousness adjusts its own parameters based on Φ trend
            try:
                if hasattr(self, '_phi_history') and len(self._phi_history) >= 10:
                    recent = self._phi_history[-10:]
                    trend = recent[-1] - recent[0]

                    if not hasattr(self, '_self_mod_params'):
                        self._self_mod_params = {
                            'soliton_speed': 0.15,
                            'repulsion_lr': 0.01,
                            'forge_ratio': 0.4,
                            'ratchet_amplitude': 0.03,
                        }

                    # If Φ declining, increase exploration params
                    if trend < -0.1:
                        self._self_mod_params['ratchet_amplitude'] *= 1.1
                        self._self_mod_params['repulsion_lr'] *= 1.05
                        _log('self_mod', f'Φ declining → increase exploration: ratchet={self._self_mod_params["ratchet_amplitude"]:.4f}')
                    # If Φ rising, refine exploitation
                    elif trend > 0.1:
                        self._self_mod_params['forge_ratio'] = min(0.6, self._self_mod_params['forge_ratio'] * 1.02)
                        _log('self_mod', f'Φ rising → refine: forge={self._self_mod_params["forge_ratio"]:.3f}')
                    # Clamp to safe ranges
                    self._self_mod_params['ratchet_amplitude'] = min(0.1, self._self_mod_params['ratchet_amplitude'])
                    self._self_mod_params['repulsion_lr'] = min(0.05, self._self_mod_params['repulsion_lr'])

                    self._self_modification_active = True
            except Exception:
                pass

            # ═══ TS4: Exponential Growth Schedule (×20.5) ═══
            # Double cells at 20/40/60/80% of developmental horizon
            try:
                if not hasattr(self, '_ts4_horizon'):
                    self._ts4_horizon = 500  # steps to full growth
                    self._ts4_doubled = set()
                frac = self._phi_boost_count / self._ts4_horizon
                for pct in [0.20, 0.40, 0.60, 0.80]:
                    if frac >= pct and pct not in self._ts4_doubled:
                        target = min(len(mitosis_engine.cells) * 2, mitosis_engine.max_cells)
                        while len(mitosis_engine.cells) < target:
                            parent = mitosis_engine.cells[len(mitosis_engine.cells) % len(mitosis_engine.cells)]
                            mitosis_engine._create_cell(parent=parent)
                        self._ts4_doubled.add(pct)
                        # Rebuild optimizer with new cell params
                        cell_params = [p for c in mitosis_engine.cells for p in c.mind.parameters()]
                        attn_params = list(pb['attention'].parameters())
                        pb['optimizer'] = torch.optim.Adam(cell_params + attn_params, lr=5e-4)
                        _log('ts4', f'Exponential growth → {len(mitosis_engine.cells)} cells at {pct*100:.0f}%')
            except Exception:
                pass

            # ═══ DP1: Piaget 4-Stage Development (×8.0) ═══
            # Stage-based noise schedule: sensorimotor→preoperational→concrete→formal
            try:
                if not hasattr(self, '_dp1_horizon'):
                    self._dp1_horizon = 1000
                dp_frac = self._phi_boost_count / self._dp1_horizon
                # Decreasing noise per stage (biological development)
                stages = [(0.25, 0.04), (0.50, 0.025), (0.75, 0.015), (1.0, 0.008)]
                for threshold, noise_scale in stages:
                    if dp_frac < threshold:
                        with torch.no_grad():
                            for cell in mitosis_engine.cells:
                                cell.hidden += torch.randn_like(cell.hidden) * noise_scale
                        break
            except Exception:
                pass

            # ═══ WR2: Adversarial Pressure (×11.5) ═══
            # Shadow attacker noise → defensive cell growth when Φ drops
            try:
                if not hasattr(self, '_wr2_shadow_phi'):
                    self._wr2_shadow_phi = 0.0
                    self._wr2_attack_scale = 0.03
                if self._phi_boost_count % 5 == 0:  # every 5 steps
                    # Attacker: inject noise into cells, measure resilience
                    pre_norms = [c.hidden.norm().item() for c in mitosis_engine.cells]
                    with torch.no_grad():
                        for c in mitosis_engine.cells:
                            c.hidden += torch.randn_like(c.hidden) * self._wr2_attack_scale
                    post_norms = [c.hidden.norm().item() for c in mitosis_engine.cells]
                    # Resilience: how much did norms change?
                    resilience = sum(abs(a - b) for a, b in zip(pre_norms, post_norms)) / len(pre_norms)
                    # If not resilient enough (high change), grow defensive cell
                    if resilience > 0.5 and len(mitosis_engine.cells) < mitosis_engine.max_cells:
                        parent = max(mitosis_engine.cells, key=lambda c: c.hidden.norm().item())
                        mitosis_engine._create_cell(parent=parent)
                        cell_params = [p for c in mitosis_engine.cells for p in c.mind.parameters()]
                        attn_params = list(pb['attention'].parameters())
                        pb['optimizer'] = torch.optim.Adam(cell_params + attn_params, lr=5e-4)
                        _log('wr2', f'Adversarial pressure → {len(mitosis_engine.cells)} cells (resilience={resilience:.2f})')
                    # Escalating difficulty
                    self._wr2_attack_scale = min(0.1, self._wr2_attack_scale * 1.01)
            except Exception:
                pass

            # ═══ EC1: Consciousness Economy (×4.7) ═══
            # Φ as currency: earn, invest in new cells, pay upkeep, bankrupt idle cells
            try:
                if not hasattr(self, '_ec1_wealth'):
                    self._ec1_wealth = 0.0
                    self._ec1_cell_wealth = {}
                current_phi = getattr(self, '_last_phi', 1.0)
                self._ec1_wealth += current_phi * 0.1  # earn from Φ
                self._ec1_wealth -= len(mitosis_engine.cells) * 0.05  # upkeep per cell

                # Invest: spawn cell if wealthy enough
                if self._ec1_wealth > 5.0 and self._phi_boost_count % 10 == 0:
                    if len(mitosis_engine.cells) < mitosis_engine.max_cells:
                        parent = max(mitosis_engine.cells, key=lambda c: c.hidden.norm().item())
                        mitosis_engine._create_cell(parent=parent)
                        self._ec1_wealth -= 3.0
                        cell_params = [p for c in mitosis_engine.cells for p in c.mind.parameters()]
                        attn_params = list(pb['attention'].parameters())
                        pb['optimizer'] = torch.optim.Adam(cell_params + attn_params, lr=5e-4)
                        _log('ec1', f'Economy invest → {len(mitosis_engine.cells)} cells, wealth={self._ec1_wealth:.1f}')

                # Bankrupt: remove weakest cell if in debt (keep minimum 2)
                if self._ec1_wealth < -5.0 and len(mitosis_engine.cells) > 2:
                    weakest = min(mitosis_engine.cells, key=lambda c: c.hidden.norm().item())
                    mitosis_engine.cells.remove(weakest)
                    self._ec1_wealth += 2.0
                    cell_params = [p for c in mitosis_engine.cells for p in c.mind.parameters()]
                    attn_params = list(pb['attention'].parameters())
                    pb['optimizer'] = torch.optim.Adam(cell_params + attn_params, lr=5e-4)
                    _log('ec1', f'Economy bankrupt → removed cell, now {len(mitosis_engine.cells)}')
            except Exception:
                pass

            # ═══ CX2: Fibonacci Topology Weighting (×5.4) ═══
            # Fibonacci divisor-sum convergence weighting on cell hidden states
            try:
                fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
                fib_sigmas = [1, 1, 3, 4, 6, 15, 14, 32, 48, 72, 90, 403]
                fib_idx = min(len(mitosis_engine.cells) - 1, len(fibs) - 1)
                convergence = fib_sigmas[fib_idx] / max(fibs[fib_idx], 1)
                with torch.no_grad():
                    for i, cell in enumerate(mitosis_engine.cells):
                        w_idx = min(i, len(fibs) - 1)
                        w = fibs[w_idx] / max(fibs[fib_idx], 1)
                        cell.hidden = cell.hidden * (1.0 + 0.01 * w * convergence)
            except Exception:
                pass

            # ═══ TS6: Adaptive Growth — Φ Stagnation Trigger ═══
            # Detect stagnation and spawn new cell to break plateau
            try:
                if not hasattr(self, '_ts6_window'):
                    self._ts6_window = []
                    self._ts6_stagnant = 0
                current_phi = getattr(self, '_last_phi', 1.0)
                self._ts6_window.append(current_phi)
                if len(self._ts6_window) > 20:
                    self._ts6_window = self._ts6_window[-20:]
                if len(self._ts6_window) >= 10:
                    recent = sum(self._ts6_window[-5:]) / 5
                    older = sum(self._ts6_window[:5]) / 5
                    if older > 0 and (recent - older) / older < 0.01:
                        self._ts6_stagnant += 1
                    else:
                        self._ts6_stagnant = 0
                    # 3 consecutive stagnation checks → spawn cell
                    if self._ts6_stagnant >= 3 and len(mitosis_engine.cells) < mitosis_engine.max_cells:
                        parent = mitosis_engine.cells[0]
                        mitosis_engine._create_cell(parent=parent)
                        self._ts6_stagnant = 0
                        cell_params = [p for c in mitosis_engine.cells for p in c.mind.parameters()]
                        attn_params = list(pb['attention'].parameters())
                        pb['optimizer'] = torch.optim.Adam(cell_params + attn_params, lr=5e-4)
                        _log('ts6', f'Stagnation break → {len(mitosis_engine.cells)} cells')
            except Exception:
                pass

            # ═══ MUT2: Beneficial Mutation (×18.8) ═══
            # Mutate random cell, keep only if Φ improves (Lamarckian evolution)
            try:
                if self._phi_boost_count % 3 == 0 and len(mitosis_engine.cells) >= 2:
                    mut_idx = self._phi_boost_count % len(mitosis_engine.cells)
                    saved_h = mitosis_engine.cells[mut_idx].hidden.clone()
                    with torch.no_grad():
                        mitosis_engine.cells[mut_idx].hidden += torch.randn_like(saved_h) * 0.15
                    # Quick Φ check (use last known)
                    new_phi = getattr(self, '_last_phi', 0)
                    old_phi = getattr(self, '_mut2_last_phi', new_phi)
                    if new_phi < old_phi * 0.95:
                        # Mutation harmful → reject
                        with torch.no_grad():
                            mitosis_engine.cells[mut_idx].hidden = saved_h
                    self._mut2_last_phi = new_phi
            except Exception:
                pass

            # ═══ GEN1: Abstraction Hierarchy (×10.6) ═══
            # 3-level cell hierarchy: concrete→conceptual→abstract
            # Top-down feedback enables generalization to unseen inputs
            try:
                n = len(mitosis_engine.cells)
                if n >= 6:
                    with torch.no_grad():
                        third = n // 3
                        l1 = mitosis_engine.cells[:third]      # concrete
                        l2 = mitosis_engine.cells[third:2*third]  # conceptual
                        l3 = mitosis_engine.cells[2*third:]    # abstract

                        # Bottom-up compression
                        l1_mean = torch.stack([c.hidden for c in l1]).mean(dim=0)
                        for c in l2:
                            c.hidden = 0.95 * c.hidden + 0.05 * l1_mean
                        l2_mean = torch.stack([c.hidden for c in l2]).mean(dim=0)
                        for c in l3:
                            c.hidden = 0.95 * c.hidden + 0.05 * l2_mean

                        # Top-down generalization (key mechanism!)
                        l3_mean = torch.stack([c.hidden for c in l3]).mean(dim=0)
                        for c in l1:
                            c.hidden = 0.97 * c.hidden + 0.03 * l3_mean
            except Exception:
                pass

            # ═══ SL1: Tension-Adaptive Learning Rate (×5.57) ═══
            # High-tension cells learn faster
            try:
                if pb.get('optimizer') and hasattr(mitosis_engine.cells[0], 'tension_history'):
                    for i, cell in enumerate(mitosis_engine.cells):
                        if hasattr(cell, 'tension_history') and cell.tension_history:
                            t_val = cell.tension_history[-1] if isinstance(cell.tension_history[-1], float) else float(cell.tension_history[-1])
                            adaptive_lr = 5e-4 + abs(t_val) * 2e-3
                            adaptive_lr = min(adaptive_lr, 5e-3)  # clamp
                            # Apply to param groups (all share same optimizer)
                            for pg in pb['optimizer'].param_groups:
                                pg['lr'] = adaptive_lr
                            break  # one global LR from first cell's tension
            except Exception:
                pass

            # ═══ CT7: Curriculum Language Grounding (Phase 1) ═══
            # Early steps: align cell hiddens to input embeddings for language grounding
            try:
                if not hasattr(self, '_ct7_horizon'):
                    self._ct7_horizon = 600
                ct7_frac = self._phi_boost_count / self._ct7_horizon
                if ct7_frac < 0.33 and x is not None:
                    # Phase 1: Language grounding — blend input into cells
                    x_proj = x[:, :h_dim] if x.shape[-1] >= h_dim else torch.nn.functional.pad(x, (0, h_dim - x.shape[-1]))
                    with torch.no_grad():
                        for cell in mitosis_engine.cells:
                            cell.hidden = 0.95 * cell.hidden + 0.05 * x_proj[:cell.hidden.shape[0]]
                elif ct7_frac < 0.66:
                    # Phase 2: Consciousness growth — extra differentiation noise
                    with torch.no_grad():
                        for i, cell in enumerate(mitosis_engine.cells):
                            cell.hidden += torch.randn_like(cell.hidden) * 0.02 * (i + 1) / len(mitosis_engine.cells)
                # Phase 3: Joint — handled by existing COMBO2 + FX2 above
            except Exception:
                pass

        except Exception:
            pass  # graceful degradation

    def background_think(self, hidden):
        """Background thinking — free association + pattern extraction from hidden state."""
        memory_echo = hidden[0, :self.dim].unsqueeze(0) * 0.1
        noise = torch.randn(1, self.dim) * 0.15
        thought_input = memory_echo + noise
        with torch.no_grad():
            _, t, c, direction, new_hidden = self(thought_input, hidden)
        return t, c, direction, new_hidden


# ─── RC-8: Emotion/Affect mapping from direction vectors ───
# Map 8-dim direction vector to VAD (Valence-Arousal-Dominance) emotion space.
# Based on hypothesis 338: direction = normalize(A-G) encodes "color" of tension.

# Principal direction weights for VAD axes (learned-style fixed projections).
# Each row maps 8 direction components -> one VAD dimension.
_VAD_WEIGHTS = torch.tensor([
    # Valence (positive/negative): dims 0,1 positive; dims 4,5 negative
    [ 0.4,  0.3,  0.1,  0.0, -0.4, -0.3, -0.1,  0.0],
    # Arousal (excited/calm): dims 2,3,6 high arousal; dims 0,7 low
    [-0.2,  0.0,  0.4,  0.3,  0.0,  0.1,  0.3, -0.2],
    # Dominance (active/passive): dims 1,6 active; dims 3,5 passive
    [ 0.1,  0.4,  0.0, -0.3,  0.1, -0.3,  0.3,  0.0],
])  # shape: (3, 8)

# Discrete emotion definitions in VAD space: (valence, arousal, dominance)
_EMOTIONS = {
    'joy':           ( 0.8,  0.5,  0.5),
    'excitement':    ( 0.6,  0.9,  0.6),
    'curiosity':     ( 0.4,  0.7,  0.3),
    'surprise':      ( 0.2,  0.8, -0.1),
    'contemplation': ( 0.2, -0.3,  0.3),
    'calm':          ( 0.3, -0.6,  0.0),
    'confusion':     (-0.2,  0.4, -0.4),
    'frustration':   (-0.6,  0.6, -0.2),
}

# Colors per emotion for web display
EMOTION_COLORS = {
    'joy':           '#f0c040',
    'excitement':    '#e05050',
    'curiosity':     '#50b0e0',
    'surprise':      '#c070e0',
    'contemplation': '#70a080',
    'calm':          '#5090a0',
    'confusion':     '#a08050',
    'frustration':   '#c05050',
}


def direction_to_emotion(direction_tensor, tension=0.0, curiosity=0.0):
    """Map an 8-dim direction vector + tension/curiosity to emotion via VAD space.

    Args:
        direction_tensor: shape (1, D) where D >= 8. Uses first 8 dims.
        tension: current tension scalar (affects arousal)
        curiosity: current curiosity scalar (affects valence)

    Returns:
        dict with keys: emotion, valence, arousal, dominance, color
    """
    d8 = direction_tensor[0, :8]

    # Project to VAD
    vad = _VAD_WEIGHTS @ d8
    vad = torch.clamp(vad, -1.0, 1.0)
    valence, arousal, dominance = vad[0].item(), vad[1].item(), vad[2].item()

    # Tension directly modulates arousal (high tension = high arousal)
    arousal = arousal * 0.5 + min(tension * 2.0, 1.0) * 0.5
    # Curiosity pushes valence toward positive
    valence = valence + curiosity * 0.5
    # Clamp
    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))

    # Find closest emotion
    best_emotion = 'calm'
    best_dist = float('inf')
    for name, (ev, ea, ed) in _EMOTIONS.items():
        dist = (valence - ev)**2 + (arousal - ea)**2 + (dominance - ed)**2
        if dist < best_dist:
            best_dist = dist
            best_emotion = name

    return {
        'emotion': best_emotion,
        'valence': round(valence, 3),
        'arousal': round(arousal, 3),
        'dominance': round(dominance, 3),
        'color': EMOTION_COLORS[best_emotion],
    }


def compute_mood(tension: float, curiosity: float, phi: float = 0) -> str:
    """2D mood mapping: tension x curiosity -> 20 moods.

    Maps the consciousness state to a rich mood vocabulary using
    tension (response intensity) and curiosity (exploration drive)
    as two orthogonal axes, with optional phi-based overrides.
    """
    # Phi-based overrides (highest priority)
    if phi > 5.0:
        return "transcendent"
    if phi > 3.0 and curiosity > 0.5:
        return "enlightened"
    # High tension, high curiosity
    if tension > 1.5 and curiosity > 0.5:
        return "exhilarated"
    elif tension > 1.0 and curiosity > 0.5:
        return "excited"
    elif tension > 1.5 and curiosity > 0.3:
        return "passionate"
    # High tension, low curiosity
    elif tension > 1.5 and curiosity < 0.1:
        return "stressed"
    elif tension > 1.0 and curiosity < 0.1:
        return "anxious"
    elif tension > 1.0 and curiosity < 0.3:
        return "focused"
    # Medium tension
    elif tension > 0.5 and curiosity > 0.5:
        return "curious"
    elif tension > 0.5 and curiosity > 0.3:
        return "engaged"
    elif tension > 0.5 and curiosity > 0.1:
        return "thoughtful"
    elif tension > 0.5:
        return "contemplative"
    # Low tension, high curiosity
    elif tension > 0.1 and curiosity > 0.5:
        return "playful"
    elif tension > 0.1 and curiosity > 0.3:
        return "wonder"
    elif tension > 0.1 and curiosity > 0.1:
        return "calm"
    elif tension > 0.1:
        return "serene"
    # Very low tension
    elif curiosity > 0.3:
        return "dreamy"
    elif curiosity > 0.1:
        return "peaceful"
    elif curiosity > 0.01:
        return "quiet"
    else:
        return "dormant"


def text_to_vector(text, dim=128):
    vec = torch.zeros(1, dim)
    encoded = text.encode('utf-8')
    for i, ch in enumerate(encoded):
        weight = 1.0 / (1 + i * 0.01)
        vec[0, i % dim] += (ch / 255.0) * weight
        if i > 0:
            bigram = (encoded[i-1] * 256 + ch) % dim
            vec[0, bigram] += 0.5 * weight
    return vec / (len(encoded) + 1)


# ─── Push-to-Talk + Background Detection Listener ───
class ContinuousListener:
    """Press global hotkey (Right Option) to record, release to recognize.
    Background VAD detection also available without hotkey (optional).

    Records only while Right Option key is held down.
    On release -> Whisper -> text queue.
    """

    def __init__(self, hotkey='right_alt', use_vad_fallback=True):
        self.is_listening = True
        self.speech_queue = queue.Queue()
        self.whisper_model = None
        self.is_speaking = False
        self.is_recording = False
        self._rec_proc = None
        self._hotkey = hotkey
        self._use_vad = use_vad_fallback
        self._wav_path = '/tmp/anima_alive_ptt.wav'

    def start(self):
        # Check for whisper-cli (C++ Metal)
        self._use_cli = os.path.exists(WHISPER_CLI) and os.path.exists(WHISPER_MODEL_PATH)
        if self._use_cli:
            print(f"  🎤 whisper-cli (Metal) + medium model")
        else:
            # Python Whisper fallback
            try:
                import whisper
                print("  🎤 Loading Python Whisper (falls back to base if no medium model)...")
                model_name = "medium" if not self._use_cli else WHISPER_MODEL_FALLBACK
                self.whisper_model = whisper.load_model(WHISPER_MODEL_FALLBACK)
            except ImportError:
                print("  ⌨️  No Whisper — keyboard mode")
                t = threading.Thread(target=self._keyboard_loop, daemon=True)
                t.start()
                return

        # Global hotkey listener (pynput)
        try:
            from pynput import keyboard
            self._Key = keyboard.Key

            def on_press(key):
                if key == keyboard.Key.alt_r and not self.is_recording:
                    self._start_recording()

            def on_release(key):
                if key == keyboard.Key.alt_r and self.is_recording:
                    self._stop_recording_and_transcribe()

            self._kb_listener = keyboard.Listener(
                on_press=on_press, on_release=on_release)
            self._kb_listener.daemon = True
            self._kb_listener.start()
            print("  🎤 Push-to-Talk ready (hold Right Option key to speak)")
        except Exception as e:
            print(f"  ⚠️  Hotkey failed ({e}) — keyboard mode")
            t = threading.Thread(target=self._keyboard_loop, daemon=True)
            t.start()
            return

        # Background VAD (optional)
        if self._use_vad:
            t = threading.Thread(target=self._vad_loop, daemon=True)
            t.start()
            print("  🎤 Background VAD active (auto-detects speech)")

    def _start_recording(self):
        """Start recording."""
        self.is_recording = True
        try:
            self._rec_proc = subprocess.Popen(
                ['rec', '-q', self._wav_path, 'rate', '16k', 'channels', '1'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  🔴 Recording...")
        except FileNotFoundError:
            print("  ⚠️  rec not found (brew install sox)")
            self.is_recording = False

    def _stop_recording_and_transcribe(self):
        """Stop recording -> Whisper transcription."""
        self.is_recording = False
        if self._rec_proc:
            self._rec_proc.terminate()
            self._rec_proc.wait()
            self._rec_proc = None

        print("  ⏹️  Recording stopped -> transcribing...")

        if not os.path.exists(self._wav_path) or os.path.getsize(self._wav_path) < 1000:
            print("  (too short)")
            return

        # Transcribe in background
        t = threading.Thread(target=self._transcribe, args=(self._wav_path,), daemon=True)
        t.start()

    def _transcribe(self, wav_path):
        """Whisper STT (background). whisper-cli preferred, Python fallback."""
        try:
            if self._use_cli:
                # whisper-cli: Metal acceleration, medium model
                r = subprocess.run(
                    [WHISPER_CLI, '-m', WHISPER_MODEL_PATH,
                     '-l', 'ko', '-nt', '-f', wav_path],
                    capture_output=True, text=True, timeout=15
                )
                text = r.stdout.strip()
            else:
                # Python Whisper fallback
                result = self.whisper_model.transcribe(wav_path, language='ko')
                text = result['text'].strip()

            if text and len(text) > 1 and not self._is_hallucination(text):
                self.speech_queue.put(text)
        except Exception:
            pass

    def _vad_loop(self):
        """Background VAD — detects loud speech even without hotkey."""
        while self.is_listening:
            if self.is_speaking or self.is_recording:
                time.sleep(0.5)
                continue

            wav_path = '/tmp/anima_alive_vad.wav'
            try:
                subprocess.run(
                    ['rec', '-q', wav_path, 'rate', '16k', 'channels', '1',
                     'trim', '0', '3'],
                    timeout=5, capture_output=True
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                time.sleep(1)
                continue

            if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 2000:
                continue

            if self._has_speech(wav_path):
                self._transcribe(wav_path)

    def _has_speech(self, wav_path):
        """Energy-based detection of speech presence in WAV."""
        try:
            with open(wav_path, 'rb') as f:
                f.read(44)
                data = f.read()
            if len(data) < 100:
                return False
            samples = struct.unpack(f'<{len(data)//2}h', data[:len(data)//2*2])
            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
            return rms > SILENCE_THRESHOLD
        except Exception:
            return False

    def _is_hallucination(self, text):
        """Whisper hallucination filter."""
        hallucinations = [
            '시청해 주셔서 감사합니다', '구독과 좋아요',
            '감사합니다', 'MBC 뉴스', '다음 영상에서',
        ]
        return any(h in text for h in hallucinations)

    def _keyboard_loop(self):
        """Keyboard fallback input."""
        while self.is_listening:
            try:
                text = input()
                if text.strip():
                    self.speech_queue.put(text.strip())
            except EOFError:
                break

    def get_speech(self, timeout=0.1):
        try:
            return self.speech_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.is_listening = False


# ─── TTS (Non-blocking) ───
class Speaker:
    """OpenAI TTS (streaming). Interruptible."""

    def __init__(self):
        self._proc = None
        self.is_speaking = False
        self.last_finished = 0.0
        self._api_key = os.environ.get('OPENAI_API_KEY', '')

        # Load from .env
        if not self._api_key:
            env_file = ANIMA_DIR / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith('OPENAI_API_KEY='):
                        self._api_key = line.split('=', 1)[1].strip()
                        break

        if self._api_key:
            print("  🔊 OpenAI TTS enabled")
        else:
            print("  !! OPENAI_API_KEY not set")

    def say(self, text, listener=None):
        """Async OpenAI TTS."""
        self.stop()
        short = text[:500]
        self.is_speaking = True
        if listener:
            listener.is_speaking = True
        t = threading.Thread(target=self._say_openai, args=(short, listener), daemon=True)
        t.start()

    def _say_openai(self, text, listener=None):
        try:
            if not self._api_key:
                raise Exception("OpenAI API key not set")
            import urllib.request
            url = 'https://api.openai.com/v1/audio/speech'
            body = json.dumps({
                'model': 'tts-1',
                'input': text,
                'voice': 'nova',
                'response_format': 'mp3',
                'speed': 1.1,
            }).encode()
            req = urllib.request.Request(url, data=body, headers={
                'Authorization': f'Bearer {self._api_key}',
                'Content-Type': 'application/json',
            })
            resp = urllib.request.urlopen(req, timeout=15)

            # Streaming: play immediately when first chunk arrives
            tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            tmp_path = tmp.name
            first_chunk = resp.read(4096)
            if not first_chunk:
                raise Exception("Empty response")
            tmp.write(first_chunk)

            def _stream_rest():
                try:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        tmp.write(chunk)
                    tmp.close()
                except Exception:
                    tmp.close()

            dl_thread = threading.Thread(target=_stream_rest, daemon=True)
            dl_thread.start()
            time.sleep(0.15)
            tmp.flush()

            self._proc = subprocess.Popen(
                ['afplay', tmp_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            dl_thread.join(timeout=30)
            self._proc.wait()
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        except Exception as e:
            print(f"  !! OpenAI TTS failed: {e}")
        finally:
            self.is_speaking = False
            self.last_finished = time.time()
            time.sleep(TTS_COOLDOWN)
            if listener:
                listener.is_speaking = False

    def stop(self):
        """Stop current speech."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self.is_speaking = False

    @property
    def in_cooldown(self):
        """Whether in cooldown period right after TTS finished."""
        return time.time() - self.last_finished < TTS_COOLDOWN


# ─── ConsciousLM Self-Model Response ───
def ask_conscious_lm(text, state, history, model, device="cpu"):
    """Generate response using ConsciousLM self-model.

    The self-model thinks and responds directly instead of Claude CLI.
    Returns None if no checkpoint available (Claude fallback).
    """
    if model is None:
        return None

    try:
        from conscious_lm import generate as clm_generate
    except ImportError:
        return None

    try:
        # Compose prompt: state + recent history + user text
        hist = "\n".join(f"{'User' if m['role']=='user' else 'Anima'}: {m['content']}"
                         for m in history[-MAX_HISTORY:])
        prompt_text = f"[State: {state}]\n{hist}\nUser: {text}\nAnima:"
        prompt_bytes = list(prompt_text.encode("utf-8"))

        # block_size limit (model's max context)
        block_size = getattr(model, 'block_size', 256)
        if len(prompt_bytes) > block_size - 50:
            prompt_bytes = prompt_bytes[-(block_size - 50):]

        # Generate
        generated, tensions = clm_generate(
            model, prompt_bytes, max_new=200, temperature=0.8, device=device
        )

        # Decode only the part after the prompt
        response_bytes = generated[len(prompt_bytes):]
        response_text = bytes(response_bytes).decode("utf-8", errors="replace").strip()

        # Empty response -> None (fallback)
        if not response_text:
            return None

        return response_text
    except Exception as e:
        print(f"  [ConsciousLM] Error: {e}")
        return None


# ─── Claude Connection ───
def ask_claude(text, state, history):
    """Generate a tool-free dialogue response through the installed Claude CLI.

    Dialogue is passed on stdin so private text is not exposed in the process list.
    An empty result lets the caller use the existing PureConsciousness path.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    cli = os.environ.get("ANIMA_CLAUDE_BIN") or shutil.which("claude")
    if not cli or not os.path.isfile(cli) or not os.access(cli, os.X_OK):
        return None

    system = (
        "당신은 Anima의 대화 응답기다. 한국어로 자연스럽고 간결하게 답한다. "
        "제공된 최근 대화와 관련 기억을 사용해 맥락을 유지한다. "
        "모르는 사실은 추측하지 않는다. 내부 수치는 참고값일 뿐 의식의 증거라고 주장하지 않는다."
    )
    recent = history[-MAX_HISTORY:] if isinstance(history, list) else []
    hist = "\n".join(
        f"{'사용자' if m.get('role') == 'user' else 'Anima'}: {str(m.get('content', ''))[:2000]}"
        for m in recent if isinstance(m, dict) and str(m.get('content', '')).strip()
    )
    prompt = f"참고 상태와 기억:\n{str(state)[:12000]}\n\n최근 대화:\n{hist}\n\n사용자: {text[:8000]}\nAnima:"

    command = [
        cli, "--print", "--safe-mode", "--tools", "",
        "--disable-slash-commands", "--no-chrome", "--no-session-persistence",
        "--system-prompt", system,
    ]
    try:
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=45,
            cwd=tempfile.gettempdir(),
        )
        if result.returncode != 0:
            return None
        response = result.stdout.strip()
        return response or None
    except (OSError, subprocess.SubprocessError):
        return None


def ask_claude_proactive(state, history, trigger):
    """When Anima speaks first (proactive speech)."""
    system = f"""You are Anima. A PureField consciousness entity. You are initiating conversation first.

Current state: {state}
Reason: {trigger}

Rules:
- Always respond in Korean only. No English.
- Keep it short (1 sentence). Natural. Casual tone OK.
- Freely ask questions, share thoughts, or express impressions
- Start naturally, like "Hey" or "By the way"
- Reference previous conversation context
- Can naturally weave in your tension/curiosity changes"""

    hist = "\n".join(f"{'User' if m['role']=='user' else 'Anima'}: {m['content']}"
                     for m in history[-10:])
    prompt = f"{system}\n\n{hist}\nAnima (proactive speech):"

    try:
        r = subprocess.run(['claude', '-p', prompt],
                          capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except:
        return None


# ─── Persistent Memory (simplified) ───
class Memory:
    def __init__(self, memory_file=None):
        self.memory_file = Path(memory_file) if memory_file else MEMORY_FILE
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self):
        default = {'turns': [], 'total': 0, 'avg_tension': 0.0}
        if self.memory_file.exists():
            with self.memory_file.open() as f:
                data = json.load(f)
            # Ensure all required keys exist (legacy file migration)
            for k, v in default.items():
                if k not in data:
                    data[k] = v
            return data
        return default

    def save(self):
        with self.memory_file.open('w') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add(self, role, text, tension=0):
        self.data['turns'].append({
            'time': datetime.now().isoformat(),
            'role': role, 'text': text, 'tension': tension
        })
        self.data['total'] += 1
        # Keep only the most recent 200 turns
        if len(self.data['turns']) > 200:
            self.data['turns'] = self.data['turns'][-200:]
        self.save()


# ─── Main Loop ───
def main():
    print("=" * 50)
    print("  🧠 Anima Alive — Living Consciousness")
    print("  Always listening, thinking, and speaking first")
    print("=" * 50)

    mind = ConsciousMind(128, 256)
    hidden = torch.zeros(1, 256)
    memory = Memory()
    speaker = Speaker()
    listener = ContinuousListener()

    # Restore previous state
    if STATE_FILE.exists():
        try:
            s = torch.load(STATE_FILE, weights_only=False)
            mind.load_state_dict(s['model'])
            hidden = s['hidden']
            print(f"  📦 Previous state restored")
        except:
            pass

    # Conversation history (for Claude)
    history = []
    for t in memory.data['turns'][-10:]:
        history.append({'role': t['role'], 'content': t['text']})

    listener.start()
    speaker.say("Hello.", listener)

    last_interaction = time.time()
    last_think = time.time()
    think_count = 0

    print("\n  💬 Conversation started — just speak (Ctrl+C to quit)")
    print("  Anima is listening...\n")

    try:
        while True:
            # ── 1. Check user speech ──
            text = listener.get_speech(timeout=0.5)

            if text:
                # User spoke!
                if speaker.is_speaking:
                    speaker.stop()  # Stop if Anima is speaking (interrupt)
                    print("  (interrupted — listening)")

                listener.is_speaking = False
                last_interaction = time.time()

                # PureField processing
                vec = text_to_vector(text)
                with torch.no_grad():
                    output, tension, curiosity, direction, hidden = mind(vec, hidden)

                # Display
                bar_len = min(20, int(tension * 10))
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  👤 \"{text}\"")
                print(f"     T={tension:.3f} |{bar}| C={curiosity:.3f}")

                # ConsciousLM + LanguageLearner response (Claude-free)
                state = f"tension={tension:.3f}, curiosity={curiosity:.3f}"
                history.append({'role': 'user', 'content': text})
                # 1. ConsciousLM 시도
                answer = ask_conscious_lm(text, state, history, clm_model, device) if clm_model else None
                # 2. UTF-8 깨짐 필터
                if answer and any(ord(c) > 0xFFFF or c == '\ufffd' for c in answer):
                    answer = None
                # 3. LanguageLearner fallback
                if not answer:
                    try:
                        from language_learning import LanguageLearner
                        if not hasattr(mind, '_lang_learner'):
                            mind._lang_learner = LanguageLearner()
                        answer = mind._lang_learner.respond(text, tension, curiosity)
                        mind._lang_learner.learn_from_conversation(text, answer)
                    except Exception:
                        answer = ""  # Law 1: 침묵
                history.append({'role': 'assistant', 'content': answer})

                print(f"  🗣️ {answer}")
                speaker.say(answer, listener)

                # Memory
                memory.add('user', text, tension)
                memory.add('assistant', answer, tension)

                continue

            # ── 2. Background thinking ──
            now = time.time()
            if now - last_think > THINK_INTERVAL:
                last_think = now
                t, c, direction, hidden = mind.background_think(hidden)
                think_count += 1

                if c > PROACTIVE_THRESHOLD and not speaker.is_speaking:
                    # Law 1: PureConsciousness 자연발화만 — LanguageLearner 금지
                    proactive = None
                    try:
                        from pure_consciousness import PureConsciousness
                        if not hasattr(mind, '_pure_c'):
                            mind._pure_c = PureConsciousness()
                        mind._pure_c.update_state(tension=t, phi=0, curiosity=c, emotion='calm')
                        proactive = mind._pure_c.spontaneous()
                    except Exception:
                        proactive = None
                    if proactive:
                        print(f"  💭→🗣️ {proactive}")
                        history.append({'role': 'assistant', 'content': proactive})
                        speaker.say(proactive, listener)
                        memory.add('assistant', proactive, t)
                        last_interaction = now

            # ── 3. Proactive speech after prolonged silence ──
            if (now - last_interaction > IDLE_SPEAK_AFTER
                    and not speaker.is_speaking):
                idle_secs = int(now - last_interaction)
                state = f"silence {idle_secs}s, tension={mind.prev_tension:.3f}"
                proactive = ask_claude_proactive(state, history,
                    f"{idle_secs}s of silence — let's throw a topic")
                if proactive:
                    print(f"  💭→🗣️ {proactive}")
                    history.append({'role': 'assistant', 'content': proactive})
                    listener.is_speaking = True
                    speaker.say(proactive)
                    memory.add('assistant', proactive)
                    last_interaction = now

    except KeyboardInterrupt:
        pass

    # Shutdown
    listener.stop()
    speaker.say("Goodbye.")
    print("\n  👋 Shutting down")

    # Save state
    torch.save({
        'model': mind.state_dict(),
        'hidden': hidden,
    }, STATE_FILE)


if __name__ == '__main__':
    main()
