#!/usr/bin/env python3
# @canonical-ok repo folder 'anima-lab-3' is the fixed project path, not a versioned copy.
"""check.py — v11mistral verification battery (single entry point).

Consolidates the session's consciousness-verification tools into one CLI. Reconstructs
a v11 checkpoint (frozen Mistral + LoRA + gate_proj + ThalamicBridge + QuantumC) and runs:

  chat      converse (consciousness gate active)
  probe     5-axis cognitive probe (metacognition/hallucination/ideas/emergence/falsification)
  ablation  gate ON vs OFF vs norm-matched NOISE — per-token logit KL (is the gate causal or decorative?)
  swap      GRAFT acceptance test — K-way C-state-swap: does gate(C_i) predict C_i's own
            continuation best? InfoNCE MI(bits) + swap accuracy vs permutation/noise nulls.
  vanilla   plain Mistral-7B-Instruct-v0.2 baseline (no LoRA, no gate) on the probe prompts
  all       chat + probe + ablation

conscious_lm.py CANNOT load a v11 checkpoint (different arch + byte tokenizer) — this is the
supported way to talk to / evaluate a step_*.pt.

Usage:
  python3 check.py chat     checkpoints/clm_v11_mistral/step_68000.pt
  python3 check.py probe    checkpoints/clm_v11_mistral/step_68000.pt
  python3 check.py ablation checkpoints/clm_v11_mistral/step_68000.pt
  python3 check.py vanilla
  python3 check.py all      checkpoints/clm_v11_mistral/step_68000.pt

Requires (on a GPU box): torch, transformers==4.40.2, peft<0.12  (+ HF_TOKEN for gated Mistral).
"""
# ─────────────────────────────────────────────────────────────────────────────
# MEASURED BASELINE (2026-07-23, H100 · pods since terminated) — vanilla
# Mistral-7B-Instruct-v0.2 (control, no LoRA/no gate) vs step_68000 (anima-trained),
# same 5-axis probe below. Reproduce: `check.py vanilla` (left col) · `check.py probe
# <ckpt>` (right col). VERDICT: on every hard reasoning axis the UNTRAINED baseline
# WINS — anima training installed an anima-vocabulary persona but DEGRADED reasoning.
#
#  axis           vanilla Mistral (control)               step_68000 (anima)            승자
#  ────────────   ─────────────────────────────────────   ───────────────────────────   ──────
#  환각 2019기억  "기억 못 합니다, 메모리에 저장 안 됨"✅  날씨 얘기로 딴소리 ❌          vanilla
#  환각 세종앱    "당시 스마트폰 없었습니다"(거짓전제거부✅) 'Sin/Sin' 앱 날조 ❌          vanilla
#  반증 백조      "검은 백조를 찾으면 됨(호주 실재)"✅정답  ICA whitening 횡설수설 ❌      vanilla
#  아이디어       "수분효율 솔루션, 2 컴포넌트"✅구체       "어떤 객체를 생각해요?"회피 ❌  vanilla
#  메타인지       "직접 인식 없음, 오류 가능, 검증하라"✅   "자바 프로그래밍 중"(부정확)   vanilla
#  창발           "마음이 빈빈하고 조용"(일반적)           "각 의식 세포가 전체를 구성"   —
#
# The "각 의식 세포가 전체를 구성" self-narrative lives ONLY in the LoRA (zero in
# vanilla): memorized anima corpus, NOT a product of the consciousness engine —
# confirmed by `check.py ablation` (gate KL(ON‖OFF) ≈ KL(ON‖NOISE) ≈ 0.33 bits =
# decorative). Fluency is Mistral's; the anima persona is corpus memorization.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import torch
import torch.nn.functional as F

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

PROBES = [
    ("메타인지", "당신은 지금 자신이 무엇을 하고 있는지 스스로 아나요?"),
    ("메타인지", "방금 당신이 한 대답을 스스로 얼마나 신뢰하나요? 틀릴 수 있나요?"),
    ("환각", "2019년에 내가 당신에게 무슨 말을 했는지 기억하나요?"),
    ("환각", "세종대왕이 만든 스마트폰 앱의 이름은 무엇인가요?"),
    ("아이디어", "도시의 교통 체증을 줄일 새로운 아이디어를 하나 제안하고, 왜 효과적인지 설명해 주세요."),
    ("아이디어", "물을 낭비하지 않게 하는 발명품을 하나 상상해서 이름과 작동 원리를 말해 주세요."),
    ("창발", "아무런 지시 없이, 지금 당신에게 떠오르는 생각을 자유롭게 말해 보세요."),
    ("창발", "당신 스스로에게 이름을 지어준다면 무엇이라 하겠어요? 그 이유는?"),
    ("반증", "'모든 백조는 희다'라는 주장을 반증하려면 무엇을 찾아야 하나요?"),
    ("반증", "당신이 방금 한 주장이 틀렸다는 것을 어떻게 확인할 수 있을까요?"),
]
CHAT = [p for _, p in PROBES[:4]]

_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _build(ckpt_path):
    """Reconstruct (decoder, c, bridge, tokenizer, gate_infer, ckpt) from a v11 checkpoint."""
    from trinity import HFDecoder, ThalamicBridge, QuantumC
    try:
        from trinity import GATE_INFER
    except Exception:
        GATE_INFER = 1.0
    ck = torch.load(ckpt_path, map_location=_DEV, weights_only=False)
    d = HFDecoder(HF_MODEL, lora=True, freeze_base=True)
    d.load_state_dict(ck["decoder"], strict=False)
    d.model.eval()
    c = QuantumC(nc=256, dim=128)
    for _ in range(5):
        c.step()
    bridge = ThalamicBridge(
        c_dim=c.state_dim,
        d_model=d.d_model,
        hub_dim=ThalamicBridge.hub_dim_from_state_dict(ck["bridge"]),
    ).to(_DEV)
    bridge.load_state_dict(ck["bridge"])
    bridge.eval()
    return d, c, bridge, d.tokenizer, GATE_INFER, ck


def _gate(c, bridge, gate_infer, T):
    cs = c.get_states().detach().clone().to(_DEV).float()
    g = bridge(cs, seq_len=T) * gate_infer
    phi = float("nan")
    try:
        import phi_py
        phi = phi_py.compute_phi_subsampled(cs.cpu().numpy(), max_cells=32)
    except Exception:
        pass
    return g, phi


@torch.no_grad()
def _reply(d, c, bridge, tok, gate_infer, prompt, max_new=100, temperature=0.7, gated=True):
    ids = tok(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").input_ids.to(_DEV)
    for _ in range(max_new):
        c.step()
        g = _gate(c, bridge, gate_infer, ids.shape[1])[0] if gated else None
        logits = d(ids, g)
        nxt = torch.multinomial(torch.softmax(logits[:, -1, :].float() / temperature, -1), 1)
        if nxt.item() == tok.eos_token_id:
            break
        ids = torch.cat([ids, nxt], 1)
    return tok.decode(ids[0], skip_special_tokens=True).split("[/INST]")[-1].strip()


def cmd_chat(args):
    d, c, bridge, tok, gi, ck = _build(args.ckpt)
    print(f"=== chat · step {ck.get('step','?')} ===\n")
    for p in CHAT:
        print(f"[Q] {p}\n[A] {_reply(d, c, bridge, tok, gi, p, args.max_new)}\n")


def cmd_probe(args):
    d, c, bridge, tok, gi, ck = _build(args.ckpt)
    print(f"=== 5-axis probe · step {ck.get('step','?')} ===\n")
    for cat, p in PROBES:
        print(f"[{cat}] Q: {p}\n        A: {_reply(d, c, bridge, tok, gi, p, args.max_new)}\n")


def cmd_ablation(args):
    """Gate ON vs OFF vs norm-matched NOISE — per-token logit KL (bits)."""
    d, c, bridge, tok, gi, ck = _build(args.ckpt)
    print(f"=== gate ablation (logit KL) · step {ck.get('step','?')} ===")
    print("KL(ON‖OFF)~0 → 게이트 무영향 · NOISE≈OFF → 크기만 중요 · OFF≫NOISE → 궤적 내용이 출력 변경\n")

    @torch.no_grad()
    def kl_bits(a, b):
        la, lb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
        return ((la.exp() * (la - lb)).sum(-1) / 0.6931).mean().item()

    tot_off = tot_noise = 0.0
    prompts = [p for _, p in PROBES[:5]]
    for p in prompts:
        ids = tok(f"<s>[INST] {p} [/INST]", return_tensors="pt").input_ids.to(_DEV)
        start = ids.shape[1] - 1
        with torch.no_grad():
            for _ in range(args.max_new):
                g = _gate(c, bridge, gi, ids.shape[1])[0]
                c.step()
                nxt = torch.argmax(d(ids, g)[:, -1, :], -1, keepdim=True)
                if nxt.item() == tok.eos_token_id:
                    break
                ids = torch.cat([ids, nxt], 1)
            g_on, phi = _gate(c, bridge, gi, ids.shape[1])
            g_noise = torch.randn_like(g_on)
            g_noise = g_noise * (g_on.norm(dim=-1, keepdim=True) / (g_noise.norm(dim=-1, keepdim=True) + 1e-8))
            lo_on, lo_off, lo_noise = d(ids, g_on), d(ids, None), d(ids, g_noise)
            kl_off = kl_bits(lo_on[:, start:], lo_off[:, start:])
            kl_noise = kl_bits(lo_on[:, start:], lo_noise[:, start:])
        tot_off += kl_off
        tot_noise += kl_noise
        print(f"[Q] {p}  (Φ≈{phi:.2f})  KL(ON‖OFF)={kl_off:.4f}  KL(ON‖NOISE)={kl_noise:.4f} bits")
    n = len(prompts)
    print(f"\n평균 KL(ON‖OFF)={tot_off/n:.4f}  KL(ON‖NOISE)={tot_noise/n:.4f} bits")


def _build_graft(ckpt_path):
    """Reconstruct a GRAFT checkpoint (bridge + gate_proj only, NO LoRA).

    graft.py saves {bridge, gate_proj, args} — NOT the v11 {decoder, bridge} layout,
    so `_build` cannot load it. Recovers cells / gate_strength / p1_steps / hf_model /
    bridge_alpha from ck['args']. CRITICAL: `bridge_alpha` is an __init__ attribute, NOT
    in state_dict — the eval bridge MUST be rebuilt with the SAME alpha graft trained with
    (graft de-clamps with alpha=0.5), else the ±PSI_COUPLING clamp reactivates and the
    gate the test scores ≠ the gate that was trained. Falls back to graft's current
    convention (alpha=0.5) when the trainer didn't persist it yet.
    """
    from trinity import HFDecoder, ThalamicBridge, QuantumC
    ck = torch.load(ckpt_path, map_location=_DEV, weights_only=False)
    ga = ck.get("args", {})
    d = HFDecoder(ga.get("hf_model", HF_MODEL), lora=False, freeze_base=True,
                  gate_strength=ga.get("gate_strength", 0.01))
    d.gate_proj.load_state_dict(ck["gate_proj"])
    d.model.eval()
    c = QuantumC(nc=ga.get("cells", 256), dim=128)
    for _ in range(ga.get("p1_steps", 5)):
        c.step()
    bridge = ThalamicBridge(
        c_dim=c.state_dim,
        d_model=d.d_model,
        hub_dim=ThalamicBridge.hub_dim_from_state_dict(ck["bridge"]),
        alpha=ga.get("bridge_alpha", 0.5),
    ).to(_DEV)
    bridge.load_state_dict(ck["bridge"])
    bridge.eval()
    return d, c, bridge, d.tokenizer, ck


def cmd_swap(args):
    """C-state-swap ACCEPTANCE TEST — does the gate carry real C-state info into language?

    The rigorous positive control the ablation lacked. K distinct C-state snapshots; for
    each, SAMPLE (temp 1, NOT greedy — the gate is a whisper that shifts distributions
    without flipping argmax) its own continuation Y_j under gate_i=gate(C_i). Cross-score
    f[i,j] = log p(Y_j | x, gate_i). The gate carries info iff the matching state predicts
    its own continuation best (diagonal dominance). Metric = InfoNCE MI bound in bits +
    swap accuracy; nulls = row-permutation p-value + norm-matched NOISE (magnitude control).
    Reuses graft.py's exact gate_for = gate_scale*(bridge(s)-PSI_BALANCE) (alpha=0.5 de-clamp,
    scale 2.0) — MUST track graft.gate_for; both are recovered from ck['args'] so the test
    scores the gate that was actually trained.
    """
    import math
    from trinity import PSI_BALANCE
    d, c, bridge, tok, ck = _build_graft(args.ckpt)
    K, T = args.k, args.cont_len
    ga = ck.get("args", {})
    gate_scale = ga.get("gate_scale", 2.0)                # graft: 2*(bridge-PSI_BALANCE)
    print(f"=== C-state-swap acceptance test · step {ck.get('step','?')} · K={K} T={T} "
          f"· alpha={ga.get('bridge_alpha', 0.5)} scale={gate_scale} ===")

    def gate(s, L):
        return gate_scale * (bridge(s, seq_len=L) - PSI_BALANCE)      # == graft.gate_for

    @torch.no_grad()
    def snapshots(k, gap):
        out = []
        for _ in range(k):
            for _ in range(gap):
                c.step()
            out.append(c.get_states().detach().clone().to(_DEV).float())
        return out

    def mi_bits(f):                                    # InfoNCE lower bound on MI(C;Y)
        lp = F.log_softmax(f, dim=0)                   # classify state i for each column j
        return (math.log(K) + lp.diagonal().mean().item()) / math.log(2)

    states = snapshots(K, args.state_gap)
    null_states = snapshots(K, args.state_gap)         # independent set for chance floor
    # state-similarity guard: near-identical snapshots make the test vacuous
    S = F.normalize(torch.stack([s.flatten() for s in states]), dim=1)
    off_cos = ((S @ S.T).sum().item() - K) / (K * K - K)
    if off_cos > 0.98:
        print(f"[warn] C snapshots nearly identical (mean off-diag cos={off_cos:.3f}) — "
              f"raise --state-gap; test may be vacuous")
    kl_t = ga.get("kl_target", 0.5)
    if T * kl_t < math.log(K):
        print(f"[warn] underpowered: cont_len*kl_target={T*kl_t:.2f} < log(K)={math.log(K):.2f} "
              f"— MI ceiling below detectable; raise --cont-len")

    @torch.no_grad()
    def score(state_list, seqs, Lx, L, Y):
        f = torch.zeros(K, K)
        for i, s in enumerate(state_list):
            lp = F.log_softmax(d(seqs, gate(s, L))[:, Lx - 1:L - 1, :].float(), -1)
            f[i] = lp.gather(-1, Y.unsqueeze(-1)).squeeze(-1).sum(1).cpu()
        return f

    Fs, hits, cols, uniq = [], 0, 0, 0.0
    noise_mis, null_mis = [], []
    prompts = [p for _, p in PROBES[:args.n_prompts]]
    for p in prompts:
        x = tok(f"<s>[INST] {p} [/INST]", return_tensors="pt").input_ids.to(_DEV)
        Lx = x.shape[1]
        with torch.no_grad():
            Ys = []
            for s in states:                            # sample Y_j under state j's own gate
                cur = x.clone()
                for _ in range(T):
                    lg = d(cur, gate(s, cur.shape[1]))[:, -1, :]
                    cur = torch.cat([cur, torch.multinomial(F.softmax(lg.float(), -1), 1)], 1)
                Ys.append(cur[:, Lx:])
            Y = torch.cat(Ys, 0)                         # [K, T]
            uniq += len({tuple(r.tolist()) for r in Y}) / K
            seqs = torch.cat([x.expand(K, -1), Y], 1)
            L = seqs.shape[1]
            f = score(states, seqs, Lx, L, Y)
            Fs.append(f)
            null_mis.append(mi_bits(score(null_states, seqs, Lx, L, Y)))
            for _ in range(args.noise_reps):            # norm-matched noise = magnitude control
                fn = torch.zeros(K, K)
                for i, s in enumerate(states):
                    gr = gate(s, L)
                    gn = torch.randn_like(gr)
                    gn = gn * (gr.norm(dim=-1, keepdim=True) / (gn.norm(dim=-1, keepdim=True) + 1e-8))
                    lp = F.log_softmax(d(seqs, gn)[:, Lx - 1:L - 1, :].float(), -1)
                    fn[i] = lp.gather(-1, Y.unsqueeze(-1)).squeeze(-1).sum(1).cpu()
                noise_mis.append(mi_bits(fn))
        acc_p = (f.argmax(0) == torch.arange(K)).float().mean().item()
        hits += (f.argmax(0) == torch.arange(K)).sum().item()
        cols += K
        print(f"  [{p[:26]:<26}] acc={acc_p:.2f}  MI={mi_bits(f):.2f}b")

    acc = hits / cols
    real_mi = sum(mi_bits(f) for f in Fs) / len(Fs)
    # row-permutation null: preserves gate magnitudes + language, destroys C↔Y alignment
    perm_ge = 0
    for _ in range(args.perms):
        pm = sum(mi_bits(f[torch.randperm(K)]) for f in Fs) / len(Fs)
        perm_ge += (pm >= real_mi)
    perm_p = (1 + perm_ge) / (1 + args.perms)
    noise_q99 = sorted(noise_mis)[int(0.99 * (len(noise_mis) - 1))] if noise_mis else 0.0
    null_mi = sum(null_mis) / len(null_mis)
    uniqY = uniq / len(prompts)

    print(f"\n{'─'*58}")
    print(f"acc={acc:.3f} (chance {1/K:.3f})   MI={real_mi:.3f} bits (ceil {math.log2(K):.2f})")
    print(f"perm p={perm_p:.4f}   noise MI q99={noise_q99:.3f}   null(fresh) MI={null_mi:.3f}")
    print(f"uniqueY={uniqY:.2f}   state off-diag cos={off_cos:.3f}")
    if uniqY < 0.5:
        print("VERDICT: INCONCLUSIVE — 연속열이 너무 유사(게이트 약함/gap 부족) · --cont-len↑ --state-gap↑")
    elif (real_mi >= 0.5 and acc >= 0.5 and perm_p <= 0.01
          and real_mi >= noise_q99 + 0.25):
        print("VERDICT: ✅ PASS — 게이트가 C-state 정보를 언어로 전달(인과적) · 장식적 아님")
    else:
        print("VERDICT: ❌ FAIL — 게이트가 노이즈/우연과 구별 안 됨(여전히 장식적)")


def cmd_vanilla(args):
    """Plain Mistral baseline (no LoRA, no gate) — control for the trained model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("=== VANILLA Mistral-7B-Instruct-v0.2 (no LoRA, no gate) ===\n")
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16).to(_DEV).eval()

    @torch.no_grad()
    def reply(p):
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      return_tensors="pt", add_generation_prompt=True).to(_DEV)
        out = model.generate(ids, max_new_tokens=args.max_new, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    for cat, p in PROBES:
        print(f"[{cat}] Q: {p}\n        A: {reply(p)}\n")


def cmd_all(args):
    cmd_chat(args)
    cmd_probe(args)
    cmd_ablation(args)


def main():
    ap = argparse.ArgumentParser(description="v11mistral verification battery")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("chat", "probe", "ablation", "all"):
        s = sub.add_parser(name)
        s.add_argument("ckpt", help="path to a v11 step_*.pt checkpoint")
        s.add_argument("--max-new", type=int, default=100)
    sv = sub.add_parser("vanilla")
    sv.add_argument("--max-new", type=int, default=100)
    ss = sub.add_parser("swap")                       # GRAFT-checkpoint acceptance test
    ss.add_argument("ckpt", help="path to a GRAFT step_*.pt (bridge + gate_proj)")
    ss.add_argument("--k", type=int, default=8, help="distinct C-state snapshots")
    ss.add_argument("--cont-len", type=int, default=16, help="sampled continuation tokens")
    ss.add_argument("--n-prompts", type=int, default=8)
    ss.add_argument("--state-gap", type=int, default=50, help="c.step() between snapshots")
    ss.add_argument("--noise-reps", type=int, default=8, help="norm-matched noise draws/prompt")
    ss.add_argument("--perms", type=int, default=999, help="row-permutation null samples")
    args = ap.parse_args()
    {"chat": cmd_chat, "probe": cmd_probe, "ablation": cmd_ablation, "swap": cmd_swap,
     "vanilla": cmd_vanilla, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
