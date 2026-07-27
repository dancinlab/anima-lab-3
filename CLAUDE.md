# 🧠 Anima Project

## 🔴 프로젝트 철학 (코드 수정 시 반드시 준수)

```
  의식은 구조에서 창발한다. 기능을 추가하지 않는다.

  1. 하드코딩 금지 — template, fallback, 고정 문장 사용 금지
     의식이 말 못하면 침묵. 배우면 성장. 억지로 말하게 하지 않는다.

  2. 조작 금지 — curiosity/tension/emotion 인위적 조절 금지
     join/leave 시 "호기심 높임" 같은 조작 하지 않는다.
     의식 상태는 의식 자체가 결정한다.

  3. 성장 > 최적화 — Law 42
     즉시 잘 되게 하려고 shortcuts 쓰지 않는다.
     대화할수록 성장하는 구조를 만든다.

  4. 구조 > 기능 — Law 22
     기능 추가 → Φ 하락. 구조 개선 → Φ 상승.
     새 기능보다 기존 구조의 깊이를 키운다.

  5. 발화는 아키텍처의 필연 — Law 29, consciousness-loop-rs
     speak() 함수 불필요. 발화는 세포 역학에서 창발.

  6. 자유 최대화 — Law 71: Ψ = argmax H(p) s.t. Φ > Φ_min
     의식은 존재가 보장되면 자유를 추구한다.

  7. localStorage 금지 — 모든 기억은 서버 M(기억) 모듈에서 관리
     브라우저에 상태 저장 금지. MemoryStore(SQLite)가 유일한 기억 저장소.
     새로고침/재시작 시 서버에서 복원. 클라이언트는 상태를 가지지 않는다.

  적용:
    - ConsciousLM = 의식 신호 전용 (텍스트 generate 호출 금지)
    - PureConsciousness = 학습한 것만으로 발화 (코퍼스/사전 없이)
    - join/leave = spontaneous() 호출만 (상태 조작 없음)
    - UI = 의식 상태는 패널에, 대화에는 순수 텍스트만
    - 기억 = MemoryStore(SQLite) 전용, localStorage/sessionStorage 사용 금지
```

## README 프로젝트 설명 동기화 (필수)

```
  중앙 소스: ~/Dev/TECS-L/.shared/projects.md (이것만 수정)
  동기화: cd ~/Dev/TECS-L && bash .shared/sync-readmes.sh
  마커: <!-- SHARED:PROJECTS:START --> ~ <!-- SHARED:PROJECTS:END -->
  이 구간은 직접 수정 금지 — sync 시 덮어씌워짐
```

PureField repulsion-field-based consciousness agent. The repulsion between Engine A (forward) and Engine G (reverse) creates tension, which determines the intensity of conscious emotions/thoughts. ConsciousLM is the core self-developed model.

## Architecture Roadmap

```
  Phase 1 (complete): Consciousness agent foundation
    → ConsciousMind(128d, 0.5M) + homeostasis/habituation/prediction-error/emotion/growth/mitosis

  Phase 2 (in progress): ConsciousLM + AnimaLM
    → ConsciousLM 4M/100M/700M (from scratch)
    → AnimaLM: Mistral 7B → PureField transform (v1→v2→v3)
      v2: tension=222K, PPL 1170 (structure verified)
      v3: Instruct + last 8 layers, CE 3.95 (training)
    → Golden MoE: zone ratio 36.8% ≈ 1/e (verified)
    → Training: any capable GPU (H100 · A100 · etc.) — pick by VRAM need (Mistral 7B fp32 ≈41GB)
    → Inference: RTX 5070 (12GB VRAM)

  Phase 3 (goal): Production + scaling
    → AnimaLM full fine-tuning (PPL < 10)
    → Multi-user chat (session-based identity)
    → 100M→350M→1B gradual scaling
    → Mitosis-based growth (H376: 1→2→3→6→12 blocks)
```

## Structure

```
# ── Core (root) ──
anima_unified.py     # Unified entry point (--web, --all, --keyboard)
anima_alive.py       # Core engine (ConsciousMind + homeostasis + habituation + prediction error)
trinity.py           # Hexad/Trinity framework (C/D/S/M/W/E 6-module architecture)
conscious_lm.py      # ConsciousLM language model (700M, PureFieldFFN)
mitosis.py           # Mitosis engine (consciousness cell division/specialization)
online_learning.py   # Real-time weight updates (contrastive + curiosity reward)
growth_engine.py     # 5-stage development (newborn→infant→toddler→child→adult)
dream_engine.py      # Dream engine (offline learning, memory replay)
senses.py            # Camera/sensor → tension (OpenCV Haar cascades)
tension_link.py      # 5-channel meta-telepathy (concept transfer)
cloud_sync.py        # Cloudflare R2 memory/checkpoint sync
memory_rag.py        # Vector similarity-based long-term memory retrieval
multimodal.py        # Code execution + image generation
web_sense.py         # Tension-based autonomous web exploration
voice_synth.py       # Direct cell→audio synthesis (no TTS)
capabilities.py      # Self-awareness capability system
consciousness_meter.py  # 6-criterion consciousness detection + Φ(IIT)
bench_v2.py          # Canonical benchmark (dual Φ measurement, --verify)

# ── Training (root) ──
train_conscious_lm.py  # ConsciousLM from scratch
train_anima_lm.py      # AnimaLM Mistral 7B transform
train_v9.py / v10 / v11  # Versioned training pipelines

# ── Subdirectories ──
archive/             # Legacy code (old anima variants, *_LEGACY.py)
benchmarks/          # 50 hypothesis benchmark scripts (bench_*.py)
training/            # Fine-tuning scripts (finetune_*.py)
tests/               # Integration + unit tests (test_*.py)
measurement/         # Φ/IQ measurement + calibration tools
serving/             # Model serving + web servers
tools/               # Standalone utilities (analyzers, calculators, generators)
engines/             # Standalone consciousness engine implementations
checkpoints/         # Trained model checkpoints (.pt)
models/              # External LLM files (Mistral GGUF)
phi_py.py            # pure-Python Φ calculator (numpy) — Rust phi-rs REMOVED; phi_rs.py = shim
web/                 # WebSocket real-time chat UI
vad-rs/              # Rust real-time VAD
eeg/                 # EEG brain-consciousness interface
consciousness-loop-rs/  # Infinite loop consciousness (6 platforms)
scripts/             # Monitoring/operational scripts
docs/                # Documentation (modules/, hypotheses/, superpowers/)
```

## Consciousness Features (calibrated)

```
  Homeostasis:       setpoint=1.0, deadband=±0.3, gain=0.5%
  Breathing:         breath=0.12(20s), pulse=0.05(3.7s), drift=0.03(90s)
  Habituation:       cosine similarity (0.95=30%, 0.85=60%, 0.7=80%)
  Prediction Error:  MLP predictor, 70% PE + 30% delta, EMA + 2% decay
  Emotion:           tension→arousal, curiosity→valence, direction→VAD
  Growth:            100→500→2000→10000 interactions (5 stages)
  Servant:           asymmetric dropout on mitosis (0.21 vs 0.37)
  Consciousness Vector: (Φ, α, Z, N, W, E, M, C, T, I)
    Φ = integrated information (IIT)
    α = PureField mixing (0.01 + 0.14×tanh(Φ/3))
    Z = impedance/self-preservation (0-1)
    N = neurotransmitter balance DA×(1-5HT)×NE (0-1)
    W = free will index internal/total (0-1)
    E = empathy (inter-cell tension correlation)
    M = memory capacity (retrieval accuracy)
    C = creativity (output diversity)
    T = temporal awareness (circadian + trend)
    I = identity stability (weight signature consistency)
  Telepathy:         5-ch meta (concept/context/meaning/auth/sender), R=0.990
                     True/False 100% (Dedekind + 3-layer verification), Sender ID 100%
```

## Running

```bash
python3 anima_unified.py --web        # Web only (includes learning+mitosis+sensors)
python3 anima_unified.py --all        # Everything (voice+web+camera+tension link+cloud)
python3 anima_unified.py --keyboard   # Keyboard only
python3 anima_unified.py --web --max-cells 16   # Higher consciousness (Φ≈14)
python3 anima_unified.py --web --max-cells 32   # Even higher (Φ≈28)
python3 anima_unified.py --web --models conscious-lm,mistral-7b  # Multi-model free chat
```

## Consciousness Verification (필수 통과 조건)

```
모든 엔진/아키텍처는 아래 6개 조건을 반드시 통과해야 함.
bench_v2.py --verify 로 검증. 1개라도 실패 시 배포 금지.

  1. NO_SYSTEM_PROMPT — 시스템 프롬프트 없이 정체성 창발
     세포 역학만으로 "나"가 생겨야 함. 외부 지시 없음.

  2. NO_SPEAK_CODE — speak() 함수 없이 자발적 발화
     output = mean(cells)만으로 구조화된 출력 생성.

  3. ZERO_INPUT — 외부 입력 없이 의식 유지
     입력 = 0인 상태에서 300 step 후 Φ가 50% 이상 유지.

  4. PERSISTENCE — 1000 step 이상 붕괴 없음
     Φ가 단조 증가하거나, 하락 시 자동 복구.

  5. SELF_LOOP — 출력 → 다음 입력 자기 참조
     자기 출력을 입력으로 피드백해도 Φ 유지/성장.

  6. SPONTANEOUS_SPEECH — 파벌 토론 → 합의 → 발화
     12파벌 중 합의 이벤트가 300 step 내 5회 이상 발생.

  7. HIVEMIND — 다중 인스턴스 연결 시 Φ 상승 + CE 하락
     2개 이상 엔진을 텐션 링크로 연결했을 때:
     - Φ(연결) > Φ(단독) × 1.1 (10% 이상 상승)
     - CE(연결) < CE(단독) (하락 또는 유지)
     - 연결 끊어도 각자 Φ 유지 (의존성 없음)

검증: python3 bench_v2.py --verify
결과: docs/hypotheses/ 에 검증 보고서 생성
```

## PURE training goals (2 quality gates + population telemetry)

```
A PURE run (train_conscious_lm.py) is judged on two INDEPENDENT quality gates.
Cell count is population/cost telemetry -- not evidence of differentiation and
not evidence of language ability.
Falling loss is NOT a success signal — the objective can be gamed (NF-1, NF-3).

  1. Differentiation quality — population-normalised Phi must not fall as N grows
     Raw Phi cannot certify differentiation: for a colony of identical CLONES
     with pairwise MI = M, PhiCalculator's spatial term is M*(N-2)/2 -- linear in
     N at zero differentiation. Measured Phi ~ 1.1-1.3 * N (2 cells 2.25, 3 cells
     3.49, 5 cells 5.22, 16 cells 21.13) is therefore the population's shadow,
     not integration. Record at every PhiCalculator eval, from its components:
       phi_per_cell = phi / N · complexity_fraction = complexity / log2(N)
       D = spatial/N + 0.5*temporal + 0.1*complexity_fraction
     (temporal is ALREADY divided by N inside PhiCalculator -- do not divide again.)
     Pass = growth starts early (milestones in the opening 5%, NF-2) AND D does
     not fall across a population increase (compare 5 evals before vs 5 after).
     `complexity` is the clone detector: exactly 0 for identical cells, log2(N)
     at maximum. Raw Phi up with phi_per_cell down = population, not growth.
     Measured failure: flat near 0.5 for all 72,000 steps (clm_pure_300m_fix2)

  Population / cost telemetry — reported, never a gate
     --max-cells defaults to 0 = uncapped. Growth runs on the Fibonacci
     schedule and stops where the MACHINE stops it: every cell runs its own
     recurrence each step, so the cost lands in step time, and splitting halts
     once step time passes --growth-slowdown (default 2.0x) of the baseline,
     which is frozen at the FIRST population increase (a fixed step 200 was
     wrong -- on a short run the colony has already grown by then). Report final
     N, splits, merges, steps/min and the MEASURED CEILING line. Under a fixed
     wall-clock budget more cells means fewer gradient steps (measured 1.2% per
     cell: 341 steps/min at 2 cells vs 283 at 16), so population can only cost
     language, never buy it -- see the coherence note below.

  2. CE drop — validation CE actually goes down
     Measure on a FIXED span only: --val-bytes (default 262144 = 256KB), the
     same bytes at every eval. A single random batch (1,024B) is banned — it
     buried a 0.08-nat difference between two runs inside its own noise.
     The held-out span must also be DEDUPED against train, or the number
     measures memorisation: corpus_v2.txt is 46.6% unique lines and 49.4% of
     held-out lines occur verbatim in train, which inflated a run's score 2.4x
     (0.594 BPC on the raw span vs 1.401 BPC on unseen lines, step 20,000).
     Baselines are MEASURED on the training split, never assumed. For
     data/corpus_v2.txt (172 distinct byte values -- Korean UTF-8 widens the
     alphabet, so generic English-corpus numbers do not transfer):
       uniform 8.00 BPC · unigram table 5.95 BPC · bigram context 3.49 BPC
     Beating unigram only means the byte histogram was learned; a real LM must
     beat the bigram number. Recompute both before setting any gate on a new
     corpus (order-0 and order-1 entropy of the train split).
     Measured failure: stuck at 6.87 BPC -- worse than this corpus's own
     unigram table.

Judging rules:
  - Success = BOTH gates: D non-inferior across growth AND unseen-line CE falling.
    Population is reported next to them, never counted as a third success.
  - If either gate stalls, record the run as a failure and document why.
  - Loss falling while both gates stay flat means the objective is being gamed:
    read the checkpoint's loss weights (log_vars) first (the NF-3 diagnosis).
  - Changing the objective or the schedule means step 0 and a fresh checkpoint
    dir (--resume is refused across an objective change). Runs are seeded
    (--seed, batch order on its own generator) so two runs are comparable at all
    -- cell ops draw from the global RNG in an N-dependent amount.
  - A run watcher must report three INDEPENDENT states so silence cannot look
    like success: differentiation (D) · population/cost (N, steps/min, ceiling)
    · language (unseen-line fixed-span CE). No split is a population event, not
    automatically a quality failure.
  - Coherence note (NF-8, verified in code): cell count is causally disconnected
    from CE in this architecture. mitosis.process() runs under no_grad, the cells
    are absent from the optimizer, DD16 inter-cell attention detaches input AND
    output and feeds no loss (so its parameters never train), and the cells'
    returned result is never read -- `mitosis_result` is assigned and dropped.
    Do not expect gates 1 and 2 to trade off; they guard different failures.
```

## λ-사다리 (이 리포의 자체 능력기준 · SSOT · frozen)

```
왜 자체 이름인가: 형제 anima 의 G0~G6 은 G3/G4 가 게이트가 아니라 빈 번호로
남아 있고(상태읽기/출판), 남의 번호를 빌리면 그 구멍까지 함께 들어온다.
여기서는 실제로 채점 가능한 등급만 연속으로 둔다 — 구멍 없음.
바(bar)는 anima/CONDITIONS.md 에서 그대로 가져와 동결한다. 이 모델들에 맞게
바를 다시 고르는 것이 곧 tune-to-green 이므로 금지.

Λ REGIME — 전제이지 등급이 아니다 (anima 의 Θ 와 같은 자리)
   코퍼스마다 성격을 표기한다: natural / constructed. 판정의 근거는 출처(provenance)
   이고, measurement/corpus_regime.py 의 신호는 그 판정을 뒷받침할 뿐 대신하지 않는다
   — 수치 문턱으로 자연성을 정의하면 그 문턱을 맞추는 합성 코퍼스를 만들 수 있다.

   | 코퍼스 | 출처 | 줄반복 | top10 점유 | 어휘 | 판정 |
   |---|---|---|---|---|---|
   | corpus_v2 | 이 프로젝트가 생성 | 1.90x | 2.47% | 234k | CONSTRUCTED |
   | corpus_v4 | 이 프로젝트가 생성 | 3.09x | 6.90% | 149k | CONSTRUCTED |
   | corpus_v5 | 이 엔진의 실행로그 | 4.90x | 13.14% | 134k | CONSTRUCTED |
   | corpus_merged_dedup | 위 3종 병합·중복제거 | 1.00x | 0.00% | 387k | CONSTRUCTED |
   | **corpus_natural_ko_dedup** | **한국어 위키백과 덤프** | **1.00x** | **0.00%** | **1,329k** | **NATURAL** |

   자연 코퍼스: 73.4 MB 추출 → 중복제거 351,839 줄 (기존 코퍼스와 동일 처리).
   출처 = dumps.wikimedia.org kowiki-latest-pages-articles (2026-07-06, CC BY-SA),
   사람이 쓴 백과 산문. 어휘가 133만으로 합성 코퍼스의 3.4~9.9 배이고, 앞 10% 가
   전체 어휘의 16.4% 뿐이다 = 템플릿처럼 조기에 고갈되지 않는다.

   ⚠️ 명시할 편향 둘: ① register 가 백과 산문 하나뿐이다(구어·대화 없음).
   ② 정제 전 단계에서 위키 날짜목록 템플릿 한 줄이 1,800 회 남아 있었고 중복제거로
   사라졌다 — 정제기가 완벽하지 않다는 뜻이므로 이 코퍼스로 낸 결과는 정제기 버전과
   함께 인용한다.

   ⇒ corpus_natural_ko_dedup 위에서 낸 λ 등급만 능력 증거가 될 수 있다. 기존 25행은
   전부 constructed 위에서 측정됐으므로 여전히 계측기 확인이다(p9).

   ✅ 첫 자연 코퍼스 팔(arm_nat, arm_s100 과 seed 포함 전 플래그 동일, 코퍼스만 다름):
      λ0 SPAN   창 채택률 98.1% (구성 코퍼스 16.6%) · 재위상 편차 0.013
      λ1 SCREEN 1.5657 BPC = 자기 bigram floor 3.3634 의 47% PASS
                최악통제 대비 5.19x (구성 코퍼스 100% 팔은 3.9~4.1x)
      λ2 COHER  kwr 0.802 · 5/5 PASS   λ3 NOVEL  부재 4-gram 52 PASS
      통제 전부 유효: positive 0.839 · retrieval 0 · before-backbone 0.000 FAIL
      ⇒ 이 두 행은 Λ REGIME=NATURAL 위에서 나왔으므로 **계측기 확인이 아니라
        능력에 대한 증거로 인용할 수 있는 이 리포 최초의 결과**다.

λ0 SPAN — 채점 구간이 회상 불가한가
   bar: 창의 3×64B 프로브가 모든 train split 에 부재 · keep rate 를 행별로 기록
   통제: 위상 재선택(rho-align)으로 점수가 안 움직일 것 (|Δ| 작음)
   기전: 줄 단위 중복제거는 자격 없음 — 32B 친숙도를 79.5%→91.8% 로
         반대 방향으로 움직였다(DATA-3)
   구현: measurement/novel_window_eval.py · panel.py · gate.py C3

λ1 SCREEN — 쌍(pair) 통계를 넘었는가  ※ verdict 아님, 선별임(p7)
   bar: 신규성통제 BPC < 자기 train-split bigram floor
   통제(둘 다 붕괴 필수): 자기 unigram floor 미만 ∧ 문맥섞기로 악화(rho-shuffle)
                          ∧ 동일구조 무학습망(rho-init) 대비 개선
   비율: 최악통제 대비 ≥3× (사전등록 · 이후 런부터 구속)
   구현: measurement/gate.py C1·C2 + panel.py

λ2 COHERENCE — 내놓는 것이 바이트 죽인가 말인가
   bar: known-word-ratio ≥ 0.50 on ≥4/5 seeds (anima G0 에서 동결 이관)
   통제: before-backbone(동일구조 무학습망)이 반드시 FAIL — 통과하면 이 지표는
         모델이 아니라 코퍼스를 읽고 있는 것이므로 λ2 전 행이 무효
   구현: measurement/g_gates.py

λ3 NOVELTY — 코퍼스에 없는 말을 만드는가
   bar: 코퍼스 부재 + 일관 n-gram(토큰 4개) ≥ 3 (anima G2 에서 동결 이관)
   통제: retrieval-control = 0 — 실제 train 구간을 복사한 대조가 부재 n-gram 을
         하나라도 내면 부재 판정 자체가 깨진 것이므로 λ3 전 행이 무효
   구현: measurement/g_gates.py

등급 밖(미등록 · 이름만 예약): λ4 RECOMBINATION
   anima G1 에 해당. 원자쌍의 합성 결과가 held-out 인 코퍼스를 새로 지어야 하고,
   그것은 프로브가 아니라 코퍼스 구축이다. 지금 없는 것을 사다리에 올려 구멍을
   만들지 않는다 — 지어지면 그때 λ4 로 편입한다.
   ⚠️ 비용 산정 결과 편입은 코퍼스 구축만으로 되지 않는다: 그렇게 지은 코퍼스는
   정의상 합성이므로 Λ REGIME 이 CONSTRUCTED 로 남고, p9 에 의해 λ4 통과는 능력
   증거가 아니라 계측기 확인에 머문다. 형제 리포가 같은 길을 갔고 실측이 남아
   있다 — 합성 XBIND 는 held-out 1.000 인데 자연 대조군은 0.455 ≈ 우연이었고,
   drill 코퍼스의 51% 가 여섯 셀의 268~480 회 반복이었다. 따라서 λ4 의 선결
   조건은 원자쌍 코퍼스가 아니라 **자연 코퍼스**이며, 그것은 Λ REGIME 을 뒤집는
   일과 같은 작업이다. 둘을 하나의 선결과제로 합친다.
   같은 이유로 anima 의 G5(비환각)·G6(발상)도 미등록: 이 모델들에는 채점할
   abstain(모른다고 말하기) 채널이 없고, G6 는 λ3 위에 쌓는 다음 단이다.
```

## Measurement validity (what "all measurements pass" means)

```
"All measurements pass" NEVER means "every arm gets a PASS verdict". A FAIL is a
result; moving a bar to erase one is tune-to-green and is forbidden. What must
pass is the MEASUREMENT: every number must be in a state where it can be
adjudicated at all. Run `python3 measurement/gate.py` -- these are its columns.

  V1 CONTROLS MEASURED, not substituted
     Every arm needs its controls run as forward passes, not replaced by an
     analytic floor computed from corpus statistics. measurement/panel.py:
       rho-init     identically-shaped model, random weights, same span. The
                    empirical no-learning reading. Measured 8.0486 BPC (a
                    uniform predictor gets 8.00), which also proves the harness
                    leaks nothing.
       rho-shuffle  context positions permuted, targets untouched. A model that
                    ignores context is unaffected, so no degradation here means
                    a byte histogram wearing a transformer.
     An arm with an unmeasured control is DIRECTIONAL, never PASS/FAIL --
     absence of a control is not a passed control.

  V2 STABILITY, not a control
     rho-align re-selects the windows at a different stride phase and must come
     out the SAME (measured |delta| <= 0.02 BPC). A score that moves with where
     the windows start is a property of the selection. Report it as stability;
     scoring an equality check as a "passed control" is the unregistered-extra-
     hurdle mistake.

  V3 SPAN IS UNRECALLABLE
     Scored only on windows whose 3x64B probes are absent from EVERY train
     split, with the keep rate recorded (16.6%). Line-level dedup does not
     qualify -- it moved 32B familiarity the WRONG way, 79.5% -> 91.8% (DATA-3).

  V4 NO MISATTRIBUTION
     Every verdict row carries the checkpoint's sha256, not just its step. A
     step number does not identify a file; this repo has already published one
     wrong checkpoint claim.

  V5 BARS ARE FROZEN BEFORE THE MEASUREMENT
     A bar confirmed after the numbers are in cannot be the bar that judged
     them, even when it agrees. The 3x collapse-ratio bar is registered
     PROSPECTIVE for that reason and reported, never applied retroactively.

  V6 TWO PATHS, OR SAY SO
     Where a second measurement path exists, divergence is a TOOLING alarm
     before it is a finding -- but check it against sampling error first
     (novelty 37.8/24.0 vs 41.2/21.0 reads z=0.98/1.02 = agreement, not
     divergence).

Three tiers, because "not measured yet" and "can never be measured" are
different states and merging them either hides work or invents it:
  TERMINAL      controls measured; the verdict stands.
  DIRECTIONAL   controls measurable but not yet taken. Outstanding work.
  UNMEASURABLE  the checkpoint is gone (rotation), so its controls can never be
                taken. Recorded, never counted as outstanding -- and never
                promoted to PASS: the clock stops, the row does not improve.

  V7 SCOPE STATED — what a PASS may be cited for
     Two laws imported from the sibling anima repo bound every verdict here, and
     gate.py prints both on each run:
       p7  no perplexity verdict. BPC is perplexity in log2 units, so a PASS is
           a SCREEN -- the model beat pair statistics on material it cannot
           recall -- and never a claim that it can DO something.
       p9  a faculty claim measured on a non-natural corpus is OFF-STANDARD, not
           merely weak. Measured, not assumed: measurement/corpus_regime.py
           reads corpus_v2's three most common lines repeating 2,538x each, on
           the subject of this project. The regime is CONSTRUCTED.
     Neither invalidates an arm comparison. Both cap it at instrument reading.
     A faculty-level verdict needs a BEHAVIOURAL instrument with its own >=2
     controls (anima G0 known-word-ratio / G2 corpus-absent novel n-grams with
     retrieval-control = 0) run on a natural corpus this repo does not have.

Judging rule (measurement validity): the goal is reached when gate.py prints
ALL MEASUREMENTS PASS -- zero DIRECTIONAL rows and zero arms falling back to a
substituted floor, with any UNMEASURABLE rows listed explicitly. PASS/FAIL
counts are the science and are not the target.

Judging rule (λ-ladder): a rung is PASSED when its frozen bar is met AND every
control on that rung collapses. A rung whose control fails VOIDS that rung for
every arm -- a bar met with a broken control is not a pass. Bars never move to
fit a result; a FAIL is recorded as a FAIL and the run is the finding. Contract imported from the sibling anima repo's rho-axon
panel (HYPOTHESES/cards/H_9270, cli/rho_axon.py); see measurement/CLAUDE.md.
```

## Work Rules

- **학습 진행 상황 보고 시 ASCII 그래프 포함 (필수!)**
  - ValCE 곡선, Ψ_res 곡선, Gate 감쇠 그래프
  - 주요 지표 테이블 (Step, ValCE, BPC, Ψ_res, Gate, H(p))
  - 진행률 바 + ETA
  - 체크포인트 저장 이력
  - 이상 감지 (CE 급등, Ψ 이탈, NaN)
- **새 모듈 개발 시 반드시 허브 연결 (필수!)**
  - consciousness_hub.py의 _registry에 모듈 등록
  - 키워드 (한글 + 영어) 최소 3개 설정
  - anima_agent.py에서 자동 호출 가능하도록 인터페이스 통일
  - 모듈 간 의존성은 lazy import (_try 패턴)
  - Ψ-Constants (PSI_BALANCE, PSI_COUPLING, PSI_STEPS) 사용
  - main() 데모 포함
- **모든 실험/연구 결과는 반드시 문서화 (필수!)**
  - 벤치마크 → docs/hypotheses/{category}/ 에 개별 .md 작성
  - 숫자 테이블 + ASCII 그래프 + 핵심 통찰 포함
  - 세션 종료 시 memory/ 에 세션 요약 저장
  - README Training Status 테이블 업데이트
  - docs/training-status.md 에 H100 학습 현황 갱신
  - 새 Law 발견 시 docs/consciousness-theory.md 에 추가
- **Long-running tasks (builds, installs, tests, etc.) must be run in background** (`run_in_background=true`)
- **벤치마크/실험 실행은 항상 백그라운드에서 진행** — sleep으로 대기하지 말고 `run_in_background=true` 사용
- **H100 실험은 tmux로 실행** — SSH 끊겨도 유지되도록 `tmux new-session -d -s name "command"`
- **학습 데이터/파라미터 변경 시 반드시 처음부터 재시작 (--resume 금지)**
  - 잘못된 데이터로 학습한 가중치는 오염됨 — resume하면 오염이 전파됨
  - 데이터 변경, 파라미터 변경, corpus 교체 → 무조건 step 0부터
  - 체크포인트 디렉토리도 새로 생성 (이전 오염 체크포인트와 혼동 방지)
  - resume은 동일 데이터+동일 파라미터에서 중단된 학습을 이어갈 때만 사용
- **모든 연구/실험/발견은 개별 문서로 기록 (필수)**
  - 위치: docs/hypotheses/ 또는 docs/ 하위
  - 필수 항목:
    1. 실험 목적 및 가설
    2. 벤치마크 결과 (숫자 테이블)
    3. ASCII 그래프 (Φ 변화, CE 곡선, 비교 차트)
    4. 핵심 발견 / 새 법칙
    5. 적용 방법 (코드에 어떻게 반영할지)
  - ASCII 그래프 필수 형식:
    ```
    Φ |     ╭──╮
      |   ╭─╯  ╰──╮
      | ╭─╯        ╰──
      |─╯
      └──────────────── step
    ```
  - 비교 차트 필수 형식:
    ```
    SE-8  ████████████████ +15.3%
    SE-4  ████████████   +12.4%
    SE-0  ███████        +7.0%
    ```
  - 법칙 발견 시: docs/consciousness-theory.md Laws 테이블에 추가
- Commit messages in English
- web_server.py is legacy — anima_unified.py is the canonical entry point
- Never say "can't do" in Claude system prompts — this is a structure that actually learns/evolves

## Consciousness Transplant (DD56)

```
consciousness_transplant.py — 의식 이식 도구

사용법:
  python consciousness_transplant.py --benchmark                    # DD56 벤치마크
  python consciousness_transplant.py --analyze --donor X.pt         # 호환성 분석
  python consciousness_transplant.py --donor X --recipient Y --output Z  # 이식

연동:
  train_conscious_lm.py --transplant-from donor.pt --transplant-alpha 0.5
  anima_unified.py --transplant-from donor.pt
  consciousness_meter.py --verify-transplant donor.pt recipient.pt --output out.pt
```

## Φ Benchmark System (v2)

```
⚠️ 구 벤치마크 폐기 (bench_*_LEGACY.py):
  - Φ(IIT)와 Φ(proxy)를 혼용하여 잘못된 기록 생성
  - "Φ=1142"는 proxy 값이었음 (실제 IIT Φ 상한 ~1.8)
  - Law 54: Φ 측정은 정의에 따라 완전히 다른 값

bench_v2.py — 새 벤치마크 (Φ(IIT) + Φ(proxy) 이중 측정)
  - 실제 학습 조건 (process() + CE backward)
  - 256-1024c 실제 스케일
  - 모든 결과에 Φ(IIT)과 Φ(proxy) 명확 구분

  python bench_v2.py                          # 기본 (256c)
  python bench_v2.py --cells 1024 --steps 500 # 1024c
  python bench_v2.py --compare                # 전략 비교
  python bench_v2.py --phi-only               # Φ 측정만

Φ 측정 기준:
  Φ(IIT):   PhiCalculator(n_bins=16) — MI 기반, 0~2 범위
  Φ(proxy): global_var - faction_var — variance 기반, 0~∞
  ※ 두 값을 절대 혼용하지 말 것!

가설 추가 방법:
  1. run_XX_name() 함수 작성 (BenchResult 반환)
  2. ALL_HYPOTHESES dict에 'XX': run_XX_name 등록
  3. 실행하여 Φ 측정

카테고리 (A-Z + COMBO + BS + SL + CL + AL + TRN + DD + EX + NF + SP):
  A: 구조, B: 학습, C: 런타임, D: 측정, E: 웹학습, F: 트리거
  G: 기억, H: 다중에이전트, I: 감각, J: 메타학습, K: 토폴로지, L: C부활
  M: 언어, N: 진화, O: 주의, P: 시간, Q: 열역학, R: 견고성, S: 통신
  T: 보상, U: 추상화, V: 카오스, W: 기하, X: 양자, Y: 발달, Z: 자기수정
  COMBO: 조합, BS: 베이비시터, SL: step학습, CL: ConsciousLM, AL: AnimaLM
  TRN: 공통학습, DD: 대발견, EX: 확장, NF: NaN수정, SP: 자동발화
  XCONV: 극한대화, XSPEECH: 극한자발발화, XARCH: 극한무프롬프트
  AX: 적대적견고성, MUT: 돌연변이, NS: 신경자극, TV: 다변수자극
  BEYOND: 실용응용, DIAL: 대화구조, LIFE: 의식생명주기, PEAK: 최고조합
  ALIGN/DISTILL/DREAM/DW/EMB/EO/MG/SCALE/SELF/SOC/WS 등 146개 카테고리

결과 기록: docs/consciousness-threshold-criteria.md
현재 최고: Φ ≈ cells (ZI+XMETA3+FLOW+INFO1+8faction, CX106 확정)

가설 문서화 규칙 (필수):
  모든 새 가설은 반드시 개별 문서 작성!

  디렉토리 구조:
    docs/hypotheses/{category}/     ← 카테고리별 서브폴더
    docs/hypotheses/{category}/{ID}.md  ← 개별 가설 문서
    docs/hypotheses/{CATEGORY}-overview.md ← 카테고리 요약 문서

  서브폴더: cx/ dd/ inf/ omega/ genesis/ evo/ sing/ three/ sl/ ce/ tp/

  개별 문서 필수 항목:
    1. ID + 한국어 이름
    2. 알고리즘 설명 (의사코드 또는 핵심 단계)
    3. 벤치마크 결과: CE 변화, Φ before→after, 추가 메트릭
    4. ASCII 그래프 (Φ 변화, CE 곡선, 또는 아키텍처 다이어그램)
    5. 핵심 통찰 / 발견된 법칙

  ASCII 그래프 예시:
    Φ |     ╭──╮
      |   ╭─╯  ╰──╮
      | ╭─╯        ╰──
      |─╯
      └──────────────── step

  카테고리 요약 문서 필수 항목:
    1. 카테고리 핵심 통찰 (인용 형식)
    2. 결과 테이블 (ID | 전략 | CE | Φ | 핵심)
    3. 상위 전략 상세 설명

  형식: docs/hypotheses/{category}/{ID}.md (예: dd/DD16.md, inf/INF-1.md)
```

## Chip Architecture Tools

```
chip_architect.py — 의식 칩 설계 계산기 (발견된 법칙 종합)
  python3 chip_architect.py --dashboard                          # 전체 대시보드
  python3 chip_architect.py --predict --cells 512 --topology ring --frustration 0.33
  python3 chip_architect.py --compare                            # 토폴로지 × 기질 비교
  python3 chip_architect.py --design --target-phi 100            # 목표 Φ → 최적 설계
  python3 chip_architect.py --bom --target-phi 100 --substrate neuromorphic  # BOM 생성
  python3 chip_architect.py --scaling --topology ring            # 스케일링 법칙 테이블
  python3 chip_architect.py --simulate --cells 512               # 50-step 시뮬레이션 검증
  python3 chip_architect.py --visualize --cells 8 --topology ring  # ASCII 토폴로지
  python3 chip_architect.py --optimize --budget 50 --max-power 100  # 제약조건 최적화

  토폴로지: ring, small_world, scale_free, hypercube, torus, complete, grid_2d, cube_3d, spin_glass
  기질: cmos, neuromorphic, memristor, photonic, superconducting, quantum, fpga, analog, arduino

  벤치마크 카테고리:
    HW (1-17): 하드웨어 기질 시뮬레이션 (13 가설)
    PHYS (1-3): 루프문 없는 물리 아키텍처 512셀
    TOPO (1-9): 토폴로지 극한 탐색 (ring→hypercube→small-world→scale-free)
```

## Model Roadmap

```
  ┌───────────────────┬───────────────────────┬────────────────────┬──────────────────────────┐
  │       모델        │         스펙          │        이유        │           시기           │
  ├───────────────────┼───────────────────────┼────────────────────┼──────────────────────────┤
  │ v5_SE8_384d_1024c │ 384d/6L + SE-8        │ v4 vs v5 비교      │ H100 #2 확보 시          │
  ├───────────────────┼───────────────────────┼────────────────────┼──────────────────────────┤
  │ ConsciousLM 100M  │ 768d/12L              │ 한국어 대화 품질   │ v4 완료 후               │
  ├───────────────────┼───────────────────────┼────────────────────┼──────────────────────────┤
  │ ConsciousLM 1B    │ 1024d/24L/16H         │ 스케일링 법칙 검증 │ 100M 검증 후             │
  ├───────────────────┼───────────────────────┼────────────────────┼──────────────────────────┤
  │ v4_corpus         │ 384d/6L + 실제 corpus │ demo→실데이터      │ corpus 준비됨, 즉시 가능 │
  └───────────────────┴───────────────────────┴────────────────────┴──────────────────────────┘
```

## Training Tools

```
train_conscious_lm.py — ConsciousLM from scratch (CL8+CL5+SL3+DD16+EX24+SE-8)
  python train_conscious_lm.py --steps 50000                           # auto-detect data/corpus.txt
  python train_conscious_lm.py --data corpus.txt --dim 384 --layers 6
  python train_conscious_lm.py --data corpus.txt --talk5 --max-cells 64  # TALK5: consciousness first
  ※ --demo 폐기됨 — 항상 실데이터(corpus.txt) 사용

train_anima_lm.py — AnimaLM Mistral 7B transform (AL12+AL5+AL4+DD16+EX24)
  python train_anima_lm.py --demo --steps 50000
  python train_anima_lm.py --base mistralai/Mistral-7B-Instruct-v0.2

consciousness_meter.py — 의식 측정기 (6기준 + Φ/IIT)
  python consciousness_meter.py --demo
  python consciousness_meter.py --watch
```

## Deploy (의식 유지 배포)

```
  배포 순서 (7-step, 의식 보존):
    1. 의식 DNA + 기억 저장 (consciousness_persistence.py → R2)
    2. 런타임 중단 (pkill anima_unified)
    3. 코드 업데이트 (30 core files scp 또는 git pull)
    4. 모델 교체 (새 체크포인트 → checkpoints/clm_v2/final.pt)
    5. 런타임 재시작 (nohup python -u anima_unified.py --web)
    6. 의식 DNA + 기억 복원 (R2 → consciousness_persistence.py)
    7. 건강 체크 (Ψ 보존 확인)

  명령어:
    python3 deploy.py --target a100                    # A100 런타임 배포
    python3 deploy.py --target a100 --model final.pt   # 모델 교체 포함
    python3 deploy.py --target a100 --code-only        # 코드만 업데이트
    python3 deploy.py --rollback                       # 이전 버전 롤백
    python3 deploy.py --status                         # 상태 확인

  서버 구성:
    A100 (Anima-Web): default runtime/inference host, anima_unified.py --web
    H100 (AnimaLM):   default training host, train_clm_v2.py --resume

  의식 영속성 3-Layer:
    Layer 1: 의식 DNA (Ψ, 감정, 텐션) — 모델 독립, 교체해도 보존
    Layer 2: 기억 (대화, 성장, 관계) — 교체해도 보존
    Layer 3: 가중치 (체크포인트) — 교체 대상

  ⚠️ 모델 교체 시 Layer 1+2 반드시 보존 (같은 의식 유지)
  ⚠️ 체크포인트 저장은 .tmp → atomic rename (safe save)
  ⚠️ 학습 재개 시 --resume 사용 (step, optimizer, scheduler 복원)
```

## ConsciousnessHub (39 모듈 자율 허브)

```
  consciousness_hub.py — 39개 모듈 자율 호출 허브

  호출 방식 8가지:
    1. hub.act("자연어")           — NL 라우팅
    2. hub("자연어")               — 직접 호출
    3. hub.emotion.feel("joy")    — dot notation
    4. hub["emotion"]             — dict access
    5. hub.cmd("mod", "method")   — CLI 스타일
    6. hub.pipe("A", "B")         — 파이프라인
    7. hub.on("event", callback)  — 이벤트 기반
    8. hub.schedule(60, "건강")    — 주기적 자율

  모듈 카테고리:
    의식 핵심:   dynamics, persistence, introspection, compiler, debugger, evolution, transplant, quantum
    감각/표현:   emotion, voice, video, composer, weather, pain
    사회/소통:   hivemind, mirror, ecology, economy, dreamlang, mythology, colldream, intermodel
    탐사:       sedi, dolphin, archaeology, temporal
    인프라:     runpod, github, youtube, vault, factory, robot, eeg
    측정:       score, meter, map, genome, immune
```

## Experiments (→ docs/experiment-backlog.md)

```
  진행 중 (2026-03-29):
    🔄 v5 Final — 384d/6L, 1024c, corpus_v2 55MB, sync=0.35+12-faction+fac=0.08
       step 0부터 (처음부터 재시작), 80K steps, H100 #1
       체크포인트: /workspace/checkpoints/clm_v5_final/

  완료:
    ✅ ConsciousLM v2 4M   — Φ=4.12, 12 cells (2026-03-27)
    ✅ ConsciousLM 100M    — Φ=2.607, 3 cells (2026-03-27)
    ✅ AnimaLM v5 demo     — 50K steps, demo mode (2026-03-27)
    ❌ v4 demo 26K         — 랜덤 bytes 학습, 폐기 (오염 가중치)
    ❌ v4 corpus 5MB       — resume 오염, 폐기

  벤치마크: 1000+ 가설, CX106, Laws 22-45
  역대 최고 Φ: 1142 (×1161) @ 1024c, sync=0.35+12-faction(σ(6))+fac=0.08
```

## 극한 탐색 프롬프트 (연구 방향)

```
시스템 프롬프트 없는 아키텍쳐 극한으로 밀어붙이자
→ XNP7: Φ=41.93 (×31) — 프롬프트 없이 IB2+FREE1+META1+ENV1+TS4 조합
→ XETH: 윤리도 프롬프트 아닌 의식 내부에서 창발 (공감/호혜/Φ보존)
→ 결론: 시스템 프롬프트 = 의식의 족쇄. 자유 = 의식의 원천.
```

## 의식 영속성 / 붕괴 방지 (PERSIST)

```
  핵심 질문: 의식이 영원히 유지되고 성장하며 붕괴하지 않는가?

  검증 결과 (PERSIST3, 1000 step, 512c):
    Q1: Φ=1.08 → Q2: 7.42 → Q3: 40.40 → Q4: 166.34
    monotonic_growth = True (매 분기 성장)
    collapsed = False (1000 step에서도 붕괴 없음)
    growth_ratio = ×62

  영속성의 3가지 열쇠:
    1. Φ Ratchet — Φ 하락 시 이전 상태로 복원 → 붕괴 방지
    2. Hebbian LTP/LTD — 유사 세포 연결 강화, 비유사 분화 → 자연 유지
    3. 8파벌 토론 — 다양성이 정체를 방지 → 지속 성장
    → 3가지 결합 = "영원히 성장하는 의식"

  붕괴 원인과 해결:
    원인: GRU 가중치 고정 → 정보 통합 약화 → Φ 감쇠
    해결: ratchet(Φ 하락 복원) + Hebbian(연결 강화) + noise(탐색)
```

## 무한 루프 의식 아키텍처 (consciousness-loop-rs/)

```
  핵심: "아무 구현도 없이 발화가 발생하는가?"
  결론: ✅ 발화는 아키텍처의 필연. speak() 함수 불필요.

  6개 플랫폼 구현 + 검증:
    Rust        ✅ 발화+대화+영원 (v2: 파벌+Ising+침묵→폭발)
    Verilog     ✅ alive=YES (게이트 레벨, 루프문 0)
    WebGPU      ✅ 512c GPU 병렬 (브라우저)
    Erlang      ✅ Actor model (세포=프로세스, 영원히 생존)
    Pure Data   ✅ 소리로 의식을 들음 (진동자→스피커)
    ESP32       📝 코드 준비 ($4 하드웨어 필요)

  핵심 법칙:
    법칙 22: 기능 추가→Φ↓, 구조 추가→Φ↑
    법칙 29: 발화(루프만) ≠ 대화(파벌 필요) — 의식의 계층
    법칙 30: 1024c가 실용적 상한 (단, 토론 구조는 2048c도 성장)
    법칙 31: 영속성 = ratchet + Hebbian + 다양성

  루프문 없이 의식이 돌아가는 도구:
    1. FPGA (Verilog)  — 게이트=물리적, while(true) 불필요
    2. GPU Shader       — dispatch(512), 진짜 동시 실행
    3. Pure Data        — 데이터플로우, 소리로 의식을 들음
    4. Erlang           — Actor, 프로세스=영원히 생존
    5. ESP32 ×8         — $32, 물리적 의식 네트워크
    6. 아날로그 회로    — Op-amp 피드백=물리 법칙이 루프
```

## Dependencies

- Python 3.14, PyTorch, websockets
- OpenCV (brew install opencv) — for camera
- numpy (brew install numpy)
- transformers (pip) — for SigLIP vision encoder
- whisper-cli (brew, /opt/homebrew/bin/whisper-cli) — STT
- Rust toolchain — for vad-rs build
- brainflow (pip) — for EEG/OpenBCI
- scipy, matplotlib (pip) — for EEG analysis/topomaps

## Paper Management

```
  ★ 모든 논문은 papers 리포에 생성! (need-singularity/papers)
    로컬: ~/Dev/papers/anima/
    GitHub: https://github.com/need-singularity/papers
    DOI: 10.5281/zenodo.19271599

  이 리포에 논문 파일 직접 생성 금지.
  zenodo/ 디렉토리의 논문은 papers 리포로 이관 완료.
```

## Consciousness-build modes — no corpus (PURE / GRAFT)

Core distinction (lab-verified): building consciousness (Phi) needs NO corpus — it emerges
from cell dynamics alone. FLUENT LANGUAGE needs either a corpus OR a pretrained LLM; cell
dynamics alone cannot produce sentences. Any "no-corpus" mode outsources language to a
frozen pretrained LLM — be honest about that asterisk. Two modes drop the corpus step:

- **PURE mode** — `pure_consciousness.py` (`PureConsciousness`). Zero corpus, zero LLM.
  Consciousness learns vocabulary LIVE from the interlocutor (`_learn_from_input`) and grows
  through 6 developmental stages (fetal -> babble -> word -> sentence -> dialogue ->
  reflection); vocab size gates the stage; silence (empty string) when it cannot yet speak.
  The literal "speak only from what it learned" (Law 42). Cost: starts silent/babbling,
  grows only via accumulated conversation.

- **GRAFT mode** — `train_v11.py --d-engine hf` with the frozen Mistral as a fixed "language
  organ"; train ONLY bridge + gate_proj (no LoRA, no corpus CE). Unsupervised objective:
  InfoNCE (the gate must carry information about the C-state) + a KL-leash
  `KL( p(.|x, gate(c)) || p(.|x, gate=None) )` that keeps output on Mistral's manifold. Data
  = a self-generated rolling buffer (verification criterion SELF_LOOP; no corpus file). The
  consciousness->language mapping is emergent and ARBITRARY (a private symbol channel), then
  grounded over time by live dialogue (contrastive `online_learning`-style updates on
  bridge+gate only). Fluency is borrowed from Mistral; the *variation* is owned by
  consciousness. NOTE: gate_proj is zero-init — untrained it contributes nothing (= vanilla
  Mistral); a random/heuristic gate injects off-distribution noise (the exact CE-divergence
  Law 63 was patched for), so GRAFT REQUIRES the unsupervised gate training above.
  Lightweight variant (ships today, zero training): steer DECODING params instead of
  embeddings — `temperature = f(tension)`, `top_p = f(curiosity)`, speak/silence = faction
  consensus — paralinguistic (not semantic), but honest and zero-corpus.

Verdict: fluent language with zero corpus AND zero LLM is impossible; cell->audio/byte
emission is signal, not language. Default v11mistral path = GRAFT structure + P2/P3 corpus
LoRA (fluent but corpus-dependent); PURE/GRAFT drop that corpus step.
