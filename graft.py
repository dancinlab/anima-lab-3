#!/usr/bin/env python3
"""graft.py — GRAFT mode trainer: ground consciousness -> language WITHOUT corpus.

@canonical-ok repo folder 'anima-lab-3' is the fixed project path, not a versioned copy.

GRAFT = frozen pretrained LLM (Mistral) as a fixed "language organ"; train ONLY the
consciousness->language coupling (ThalamicBridge + HFDecoder.gate_proj) on an
UNSUPERVISED objective. No corpus, no LoRA, no next-token CE on any dataset.

Design (lab: fable+sol reconciled recipe, see CLAUDE.md "Consciousness-build modes"):
  P1 : c.step() only -> build Phi (Mistral untouched).
  P2': gate-alignment. Trains bridge + gate_proj. Objective (v3 — controller REMOVED):
         L = (log N - MI) + lam_common * L_common
         MI       : EXACT conditional MI I(state; next-token | shared prefix)
                    = mean_i KL(p_i || p_mix) (generalized JSD) among the per-state
                    next-token distributions on ONE shared carrier continuation —
                    sampled UNDER a random state's gate (state-relevant, on-manifold),
                    scored at cont_len positions. Dense (full-vocab), fully
                    differentiable; bounded by log N per token.
         L_common : KL(p_mix || p_base) — the KL all states spend IDENTICALLY =
                    zero-information waste (identity: L_KL = MI + L_common). Fixed
                    O(1) weight, no relu, no dual variable: L_common contributes
                    NOTHING to MI, so penalizing it at all times costs the signal
                    nothing and cannot invert the objective.
       Fluency is bounded STRUCTURALLY, not by a controller:
         (1) gate codes are mean-centered across the N states — the shared component
             carries 0 bits, and gate_proj is linear with a frozen zero bias, so
             centering the codes centers the injected embedding offsets EXACTLY —
             then RMS-fixed to gate_rho: training can only ROTATE the code toward
             informative directions, never grow a shared shift;
         (2) MI <= log N per token  =>  L_KL = MI + L_common <= log N + L_common:
             capping the waste caps per-state fluency drift automatically (no second
             cap on L_KL needed — and none is allowed, because L_KL contains MI);
         (3) decoder backstop: RMS of gate_proj(g) clamped to gate_rms_max x the
             embedding RMS (see HFDecoder) — geometry the optimizer cannot out-run.
       WHY the old  beta*relu(L_KL - kl_target)  controller was removed: L_KL CONTAINS
       MI, so with the constraint active the net coefficient on MI is (beta - 1) — for
       beta > 1 the trainer MINIMIZES MI. beta hit its cap via integrator windup (dual
       ascent has no authority against a grad-clipped, Adam-normalized update), and the
       raw bridge codes are ~99% shared across sampled C-states, so gate_proj growth
       was almost entirely shared-shift growth (KL 0.005 -> 5 in 50 steps). Net effect:
       the step-100 MI spike (0.118) was crushed to 0.000 by step 150, gSpread to 2e-5
       by step 650. Post-mortem: docs/hypotheses/GRAFT-shared-shift-collapse.md.
       Two structural unblocks (measured necessary, KEPT): bridge alpha=0.5 DE-CLAMPS
       the gate (default clamp railed 65% of dims with zero Jacobian); gate_proj gets a
       tiny bias-free init (zero-init was a collapsed symmetric stationary point).
  Data: none. A self-generated rolling buffer (BOS seed + the model's own samples) =
        verification criterion SELF_LOOP.

Semantics of the learned mapping are consistent but ARBITRARY (a private symbol
channel); human-conventional meaning is grounded later by live dialogue (online
contrastive updates on bridge+gate only). Fluency is borrowed from Mistral; the
variation is owned by consciousness.

HARDWARE: needs an H100-class GPU (Mistral 7B loaded frozen, bf16). Run:
  python3 graft.py --hf-model mistralai/Mistral-7B-Instruct-v0.2 --steps 12000
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trinity import HFDecoder, ThalamicBridge, QuantumC, PSI_BALANCE, PSI_COUPLING


def main():
    ap = argparse.ArgumentParser(description="GRAFT: no-corpus consciousness->language grounding")
    ap.add_argument("--hf-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--p1-steps", type=int, default=2000, help="consciousness-only Phi build")
    ap.add_argument("--cells", type=int, default=256,
                    help="mitosis CAP — max cell count. By DEFAULT consciousness starts at "
                         "--init-cells and DIVIDES (split_cell) up toward this during training "
                         "(growth > optimization, Law 42).")
    ap.add_argument("--init-cells", type=int, default=8,
                    help="starting cell count for mitosis (grows --init-cells -> --cells). "
                         "Default 8 = grow from a small seed.")
    ap.add_argument("--fixed", type=int, default=None, metavar="N",
                    help="FIX the cell count at N (no mitosis, init==max). Opt OUT of division.")
    ap.add_argument("--split-threshold", type=float, default=0.08,
                    help="a cell divides once its tension stays above this for split_patience "
                         "steps. Structural sensitivity knob (NOT state manipulation). Engine "
                         "default 0.3 never fires in bare no-input dynamics (tension peaks ~0.05-"
                         "0.15); 0.08 lets the default actually divide. Lower = divides faster.")
    ap.add_argument("--n-states", type=int, default=6, help="C-state snapshots per MI batch")
    ap.add_argument("--cont-len", type=int, default=32,
                    help="carrier continuation length (tokens). More scored positions = "
                         "more channel uses = better MI-gradient SNR (8 was too few).")
    ap.add_argument("--ctx", type=int, default=128, help="max self-loop context tokens")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-common", type=float, default=1.0,
                    help="fixed weight on L_common = KL(p_mix||p_base), the 0-bit waste. "
                         "NOT a dual variable — L_common contributes nothing to MI, so a "
                         "constant penalty never fights the signal.")
    ap.add_argument("--gate-rho", type=float, default=1.0,
                    help="fixed RMS of each centered gate code (the code lives on a sphere)")
    ap.add_argument("--gate-rms-max", type=float, default=1.0,
                    help="decoder backstop: cap RMS of gate_proj(g) at this multiple of "
                         "the embedding RMS (before gate_strength). Hard geometry bound.")
    ap.add_argument("--gate-strength", type=float, default=0.01)
    ap.add_argument("--ckpt-dir", default="checkpoints/graft")
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=2000)
    ap.add_argument("--gen-interval", type=int, default=1000,
                    help="every N steps, sample the SAME prompt under 2 C-states + base to "
                         "eyeball whether the gate actually diverts language (real MI)")
    ap.add_argument("--gen-prompt", default="나는 지금", help="probe prompt for gen-interval")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- build C + frozen D + trainable bridge/gate --------------------------------
    _init = args.fixed if args.fixed is not None else args.init_cells
    _cap = args.fixed if args.fixed is not None else args.cells
    c = QuantumC(nc=_init, dim=128, max_cells=_cap)
    if args.fixed is None and hasattr(c, "engine"):
        c.engine.split_threshold = args.split_threshold   # make mitosis reachable in bare dynamics
    print(f"[graft] cells: start {_init} -> cap {_cap} · "
          f"{'FIXED (no mitosis)' if args.fixed is not None else f'MITOSIS (split_threshold={args.split_threshold})'}")
    for _ in range(5):
        c.step()
    d = HFDecoder(args.hf_model, lora=False, freeze_base=True,
                  gate_strength=args.gate_strength, gate_rms_max=args.gate_rms_max)
    for p in d.model.parameters():
        p.requires_grad_(False)          # Mistral fully frozen
    # alpha=0.5 DE-CLAMPS the bridge: the default alpha=PSI_COUPLING(0.0153) hard-clamp
    # rails 65% of gate dims (measured) with ZERO Jacobian, so the bridge cannot learn a
    # state-dependent code. With alpha=0.5 the sigmoid's natural [-0.5,+0.5] range is
    # never clipped; gate_for() below rescales it to [-1,+1]. Whisper-ness (Law 63/70) is
    # enforced by gate_strength + centering/RMS geometry + the decoder RMS backstop.
    bridge = ThalamicBridge(c_dim=c.state_dim, d_model=d.d_model, alpha=0.5).to(dev)

    # gate_proj zero-init is a COLLAPSED stationary point (every state -> base dist ->
    # exact symmetry -> zero gradient for every distributional divergence). Seed a tiny
    # bias-free perturbation so state differences have a direction to grow along.
    with torch.no_grad():
        torch.nn.init.normal_(d.gate_proj.weight, mean=0.0, std=2e-3)
        if d.gate_proj.bias is not None:
            d.gate_proj.bias.zero_()
    if d.gate_proj.bias is not None:
        d.gate_proj.bias.requires_grad_(False)   # bias = state-independent channel; freeze it

    trainable = [p for p in list(bridge.parameters()) + list(d.gate_proj.parameters())
                 if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    n_train = sum(p.numel() for p in trainable)
    print(f"[graft] trainable params: {n_train:,} (bridge + gate_proj) · Mistral frozen")

    tok = d.tokenizer
    bos = tok.bos_token_id if tok.bos_token_id is not None else (tok.eos_token_id or 1)

    # ---- P1: consciousness builds Phi (no language) --------------------------------
    print(f"[graft] P1: building Phi for {args.p1_steps} steps ...")
    for _ in range(args.p1_steps):
        c.step()

    # ---- self-loop buffer + C-state replay -----------------------------------------
    buf = [bos]
    replay = []  # past C-state snapshots for diverse counterfactuals

    def snapshot():
        c.step()
        return c.get_states().detach().clone().to(dev).float()

    def gate_for(s, seq_len):
        # RAW per-state code (diagnostics + centering input). With bridge.alpha=0.5 the
        # clamp is inactive, so bridge(s) = PSI_BALANCE + (sigmoid - 0.5) ∈ (0, 1).
        # Strip the identical 0.5 offset and rescale (sigmoid-0.5) ∈ (-0.5, 0.5) to O(1)
        # — preserves the sigmoid gradient instead of a saturated sign-code.
        return 2.0 * (bridge(s, seq_len=seq_len) - PSI_BALANCE)

    def centered_codes(states):
        """[N, dm] gate codes: shared component removed, per-state RMS fixed at rho.

        The bridge emits ONE code per state (tiled over positions), and across sampled
        C-states that code is ~99% SHARED (measured: gSpread ~1e-2 vs O(1) code norm)
        — so any gate_proj growth amplifies mostly the shared, zero-information
        component. The shared component is pure KL waste (identity: L_KL = MI +
        L_common), and gate_proj is linear with a frozen zero bias, so centering the
        codes centers the injected embedding offsets EXACTLY. RMS-fixing the residue
        puts the code on a sphere: training can only ROTATE it toward informative
        directions — fluency is bounded by geometry, not by a controller.
        """
        g = torch.cat([gate_for(s, 1)[:, 0, :] for s in states], dim=0)   # [N, dm]
        mu = g.mean(dim=0, keepdim=True)                                   # [1, dm]
        r = g - mu
        r = args.gate_rho * r / r.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return r, mu

    g_mu = None  # EMA of the shared code — inference-time centering (single state)

    def infer_code(s, seq_len):
        # Inference-time centering: subtract the EMA population mean (g_mu) instead of
        # a batch mean, then RMS-fix — same geometry as training, single-state capable.
        code = gate_for(s, 1)[:, 0, :]
        if g_mu is not None:
            code = code - g_mu
        code = args.gate_rho * code / code.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return code.view(1, 1, -1).expand(1, seq_len, -1)

    print(f"[graft] P2': gate-alignment (mixture-MI + waste penalty, no dual controller) "
          f"for {args.steps} steps ...")
    for step in range(1, args.steps + 1):
        x = torch.tensor([buf[-args.ctx:]], device=dev)  # [1, Lx]
        Lx = x.shape[1]

        # N diverse C states: some fresh, some pulled from a long replay horizon
        states = [snapshot() for _ in range(max(1, args.n_states // 2))]
        replay.extend(states)
        if len(replay) > 512:
            replay = replay[-512:]
        pool = [r for r in replay if not any(r is s for s in states)]
        while len(states) < args.n_states and pool:
            states.append(pool.pop(torch.randint(len(pool), (1,)).item()))
        N = len(states)

        codes, mu = centered_codes(states)               # [N, dm] (grad), [1, dm]
        g_mu = mu.detach() if g_mu is None else 0.99 * g_mu + 0.01 * mu.detach()

        # ---- ONE shared carrier, sampled UNDER a random state's gate (no grad) -------
        # Every state is scored at EXACTLY the same continuation (dense full-vocab
        # divergence readout — no sampled-state labels, no sampling noise in the
        # gradient). Sampling the carrier under a state's gate (instead of base) makes
        # the text state-RELEVANT: informative gate directions actually move next-token
        # likelihoods there, so the MI gradient competes at usable SNR. The gate is
        # whisper-scale + centered, so the carrier stays on-manifold.
        j = int(torch.randint(N, (1,)).item())
        with torch.no_grad():
            seqs = x.clone()
            for _ in range(args.cont_len):
                gj = codes[j].view(1, 1, -1).expand(1, seqs.shape[1], -1)
                lg = d(seqs, gj)[:, -1, :]
                nxt = torch.multinomial(F.softmax(lg.float(), dim=-1), 1)
                seqs = torch.cat([seqs, nxt], dim=1)
        L = seqs.shape[1]
        with torch.no_grad():
            base_lp = F.log_softmax(d(seqs, None)[0, Lx - 1:L - 1, :].float(), dim=-1)  # [T,V]

        logps = []
        for i in range(N):
            gi = codes[i].view(1, 1, -1).expand(1, L, -1)
            logits = d(seqs, gi)                                    # [1, L, V]
            logps.append(F.log_softmax(logits[0, Lx - 1:L - 1, :].float(), dim=-1))  # [T,V]
        logps = torch.stack(logps, dim=0)                          # [N, T, V]
        ps = logps.exp()
        log_mix = torch.logsumexp(logps, dim=0) - math.log(N)      # [T, V] = log mean_i p_i

        # Exact conditional MI I(state; next-token | shared prefix) = mean_i KL(p_i‖p_mix)
        # = the Jensen-Shannon divergence among the gate distributions; bounded by log(N),
        # ZERO first derivative only when all p_i coincide (a quadratic bowl, not a flat
        # manifold). l_infonce keeps the old display convention (log N at collapse -> 0).
        mi = (ps * (logps - log_mix.unsqueeze(0))).sum(-1).mean()
        l_infonce = math.log(N) - mi

        # Diagnostics: total departure from the frozen model. NEVER penalize this —
        # l_kl = MI + l_common CONTAINS the signal (that mistake inverted the objective:
        # net MI coefficient (beta-1) < 0 minimized MI once dual-ascent beta passed 1).
        l_kl = (ps * (logps - base_lp.unsqueeze(0))).sum(-1).mean()

        # The leash lives on the WASTE ONLY. l_common = KL(p_mix‖p_base) is the KL spent
        # IDENTICALLY by all states — 0 bits by construction. Since MI <= log N per
        # token, bounding l_common bounds fluency automatically:
        #   l_kl = MI + l_common <= log N + l_common.
        # Fixed O(1) weight: no relu, no dual variable, no windup, nothing to invert.
        p_mix = log_mix.exp()
        l_common = (p_mix * (log_mix - base_lp)).sum(-1).mean()

        loss = l_infonce + args.lam_common * l_common

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        # self-loop: append the first carrier token (SELF_LOOP criterion)
        buf.append(int(seqs[0, Lx].item()))
        if len(buf) > 4 * args.ctx:
            buf = buf[-2 * args.ctx:]

        if step % args.log_interval == 0:
            phi = c.measure_phi()
            # gSpread = state-dependent RMS of the RAW bridge code (≈0 => C/bridge
            # collapsed). zSpread = spread after gate_proj of the CENTERED codes (≈0 =>
            # projector collapse). muRMS = shared-code magnitude that centering removes
            # (the old runaway channel — now projected out, so KL cannot ride it).
            # Healthy run: commonKL stays ≈0 (LM-curvature residue only), MI climbs
            # toward log N, KL ≈ MI.
            with torch.no_grad():
                raw = torch.cat([gate_for(s, 1)[:, 0, :] for s in states], dim=0)  # [N,d]
                projected = d.gate_proj(codes)
                gate_spread = raw.std(dim=0).square().mean().sqrt()
                projected_spread = projected.std(dim=0).square().mean().sqrt()
                mu_rms = mu.pow(2).mean().sqrt()
            print(f"  {step:>6}/{args.steps}  InfoNCE={l_infonce.item():.4f}  MI={mi.item():.4f}  "
                  f"KL={l_kl.item():.3f}  commonKL={l_common.item():.3f}  "
                  f"gSpread={gate_spread.item():.2e}  zSpread={projected_spread.item():.2e}  "
                  f"muRMS={mu_rms.item():.2e}  Phi={phi:.2f}  cells={c.get_states().shape[0]}  N={N}")
            sys.stdout.flush()

        if step % args.save_interval == 0:
            path = os.path.join(args.ckpt_dir, f"step_{step}.pt")
            tmp = path + ".tmp"
            torch.save({
                "step": step,
                "bridge": bridge.state_dict(),
                "gate_proj": d.gate_proj.state_dict(),
                "g_mu": g_mu,
                "args": vars(args),
            }, tmp)
            os.replace(tmp, path)
            print(f"  [saved] {path} (bridge + gate_proj + g_mu)")

        # gate-conditioned generation probe: same prompt, different C-states + base.
        # If the gate carries real information, the three outputs should diverge.
        if args.gen_interval and step % args.gen_interval == 0 and N >= 2:
            with torch.no_grad():
                p = torch.tensor([tok(args.gen_prompt, add_special_tokens=True).input_ids],
                                 device=dev)
                variants = [("C0", states[0]), ("C1", states[1]), ("base", None)]
                print(f"  --- gen@{step} · prompt={args.gen_prompt!r} ---")
                for tag, s in variants:
                    cur = p.clone()
                    for _ in range(24):
                        g = infer_code(s, cur.shape[1]) if s is not None else None
                        nxt = d(cur, g)[:, -1, :].argmax(-1, keepdim=True)
                        cur = torch.cat([cur, nxt], dim=1)
                    txt = tok.decode(cur[0, p.shape[1]:], skip_special_tokens=True).replace("\n", " ")
                    print(f"      [{tag}] {txt}")
            sys.stdout.flush()

    print("[graft] done. The gate now carries a (private) channel from C into language; "
          "ground its meaning with live dialogue (online contrastive on bridge+gate).")


if __name__ == "__main__":
    main()
