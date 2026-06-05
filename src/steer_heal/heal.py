"""Q1 heal: train one round's LoRA = SFT(kept completions) + divergence-to-original barrier.

The barrier reference is the round-0 ORIGINAL (gates/adapters off), not the
previous student, so it resists cumulative drift. reg picks the divergence:
  nll     SFT only (control)
  kl_fwd  KL(orig || theta)  mass-covering (dilutes the trait)
  kl_rev  KL(theta || orig)  mode-seeking (suppresses low-orig-prob = incoherent)  [expected best]
  wd      weight decay on the adapter only
"""

import random

import torch
from loguru import logger
from torch.nn import functional as F
from tqdm import tqdm
from transformers import BatchEncoding, get_cosine_schedule_with_warmup

from steer_heal.config import RunConfig
from steer_heal.ws.adapter import ModulatedLoRA
from steer_heal.ws.bake import AdapterSpec, baked


def _kl_per_pos(logp_a, logp_b):  # KL(a || b) summed over vocab, per position
    return (logp_a.exp() * (logp_a - logp_b)).sum(-1)


def _spectral_div(lora, n_iter: int = 3) -> torch.Tensor:
    """Mean operator norm σ_max(ΔW) over the adapter's layers, ΔW = (alpha/r)·B@A.

    Power iteration (u,v held constant) gives σ_max = uᵀ(B@A)v, differentiable in A,B.
    This is the weights-space analog of weight_decay: wd penalises ||ΔW||_F (sum of all
    singular values squared), spectral_norm penalises ||ΔW||_2 (the LARGEST singular value),
    i.e. it caps how much the update can stretch any single input direction. Used with tau=0
    so relu(div-0)=div is an always-on penalty (like wd), not a hinge barrier."""
    scale = lora.cfg.alpha / lora.cfg.r
    sigmas = []
    for name in lora.A:
        A, B = lora.A[name].float(), lora.B[name].float()  # A: r×d_in, B: d_out×r
        with torch.no_grad():
            v = torch.randn(A.shape[1], device=A.device)
            v = v / v.norm()
            for _ in range(n_iter):
                u = B @ (A @ v); u = u / (u.norm() + 1e-8)
                v = A.T @ (B.T @ u); v = v / (v.norm() + 1e-8)
        sigmas.append(scale * (u @ (B @ (A @ v))))  # u,v const -> grad flows through A,B
    return torch.stack(sigmas).mean()


def _gnorm(grads) -> float:  # L2 norm of a flat concat of (possibly None) param grads
    sq = sum(float(g.pow(2).sum()) for g in grads if g is not None)
    return sq ** 0.5


def _encode(tok, prompt: str, completion: str, max_len: int, device):
    # Tokenize prompt and completion SEPARATELY then concatenate the ids, so the prompt is always a
    # clean token-prefix -- no BPE merge can span the boundary (which would silently shift the SFT
    # mask by a token). prompt keeps generation's tokenization (add_special_tokens default, matching
    # generate_steered's tok(text)); the completion adds no specials. Truncation keeps the FRONT.
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0]
    comp_ids = tok(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
    input_ids = torch.cat([prompt_ids, comp_ids])[:max_len].unsqueeze(0).to(device)
    n_prompt = prompt_ids.shape[0]
    L = input_ids.shape[1]
    ids = BatchEncoding({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
    tgt_is_completion = torch.arange(1, L, device=device) >= n_prompt  # mask over next-token targets
    return ids, tgt_is_completion


def _val_nll(model, tok, val_kept, hist_specs, lora, cfg) -> float:
    """Held-out SFT nll (same student state as train: history baked, adapter live). The trait
    eval is the real metric, but val_nll catches the optimisation failure modes the eval can't:
    train falls + val rises = overfit; NEITHER falls = data near-base / opt broken."""
    if not val_kept:
        return float("nan")
    losses = []
    with torch.no_grad(), baked(model, hist_specs), lora(model, c=1.0):
        for c in val_kept:
            ids, mask = _encode(tok, c["prompt"], c["completion"], cfg.max_len, model.device)
            if mask.sum() == 0:
                continue
            logp = model(**ids).logits[0, :-1].log_softmax(-1)
            losses.append(F.nll_loss(logp[mask], ids.input_ids[0, 1:][mask]).item())
    return sum(losses) / len(losses) if losses else float("nan")


def heal_round(model, tok, kept: list[dict], hist_specs: list[AdapterSpec], cfg: RunConfig):
    """Train a fresh round adapter on top of baked history. Returns (lora, spec)."""
    assert len(kept) >= cfg.min_train, (
        f"only {len(kept)} kept completions; need >= {cfg.min_train} to train. The steering/filter "
        "starved the data (over-steered -> all garbage, or ppl_tau too strict). Fix upstream, do not train."
    )
    # hold out ~1/8 for a val nll curve (shuffled so val isn't all one alpha). Tiny-dev keeps all
    # for train (len//8 == 0) so the path still runs.
    shuf = kept[:]
    random.Random(cfg.seed).shuffle(shuf)
    n_val = len(shuf) // 8
    val_kept, train_kept = shuf[:n_val], shuf[n_val:]
    lora = ModulatedLoRA(model, r=cfg.lora_r, alpha=cfg.lora_alpha, layer_range=cfg.layer_range)
    params = list(lora.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.adam_betas,
                            weight_decay=cfg.weight_decay)
    n_steps = len(train_kept) * cfg.epochs
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=int(cfg.warmup_ratio * n_steps), num_training_steps=n_steps)

    # round-ramped barrier (config.lam_round_pow): round index = len(hist_specs) (R adapters baked = round R).
    # lam_round_pow=0 -> lam_eff==lam (constant, no behaviour change). >0 grows the barrier with round.
    rnd = len(hist_specs)
    lam_eff = cfg.lam * (1 + rnd) ** cfg.lam_round_pow

    # streaming training table (token-efficient-logging): one row, columns self-decode below.
    logger.info(f"heal[{cfg.reg}] {len(train_kept)} train (+{len(val_kept)} val) x {cfg.epochs} ep = {n_steps} steps; "
                f"lora r={cfg.lora_r} a={cfg.lora_alpha} on layers {cfg.layer_range}; "
                f"lr={cfg.lr} cosine warmup={cfg.warmup_ratio} betas={cfg.adam_betas}; "
                f"lam_eff={lam_eff:.3f} (lam {cfg.lam} x (1+round={rnd})^{cfg.lam_round_pow})")
    logger.info("SHOULD (val): train_nll falls each epoch (SFT fits the kept data); val_nll falls then "
                "flattens. If val_nll RISES while train falls -> overfit (fewer epochs / lower r). If "
                "NEITHER falls -> data is near-base (nothing to distil) or the optimiser is broken.")
    logger.info(f"SHOULD: nll (SFT) falls as the adapter learns the trait; kl (barrier div) is 0 for "
                f"reg=nll/wd and >0 for kl_rev/kl_fwd; gnorm finite (not exploding). loss = nll + lam*relu(kl-tau). "
                f"If kl stays < tau={cfg.tau} the barrier NEVER fired and {cfg.reg} == nll (no regularisation).")
    logger.info(
        "SHOULD (barrier balance): g_bar/g_nll is the gradient-pressure ratio (||∇barrier|| / ||∇sft||). "
        ">>1 -> barrier dominates, it is undoing the trait the SFT installs (over-tight: lower lam or raise tau); "
        "~1 -> balanced; 0 -> barrier inert (kl<tau, or reg=nll/wd where decay acts in the optimiser, not the loss)."
    )
    logger.info("  step   nll↓    kl  g_nll  g_bar  g_bar/g_nll  loss↓  gnorm        lr")
    # init val nll BEFORE any training = the baseline; epoch val nlls only mean something against it.
    # With B=0 the fresh adapter is a no-op, so this should equal the base model's nll on val.
    logger.info(f"  epoch init: train_nll=  nan  val_nll={_val_nll(model, tok, val_kept, hist_specs, lora, cfg):.3f}  lr={sched.get_last_lr()[0]:.1e}")
    pbar = tqdm(total=n_steps, desc=f"heal[{cfg.reg}]", mininterval=120, maxinterval=120)
    step = 0
    nlls = []  # per-step SFT loss; final = mean of last 5, the heal-stage number for the round table
    for ep in range(cfg.epochs):
        ep_nlls = []
        for c in train_kept:
            ids, mask = _encode(tok, c["prompt"], c["completion"], cfg.max_len, model.device)
            if mask.sum() == 0:
                # prompt filled max_len so the completion was truncated to zero target tokens.
                # Loud, not silent: this is a kept completion lost from training (review).
                logger.warning(f"heal: 0 target tokens (prompt >= max_len={cfg.max_len}), skipping a kept completion")
                pbar.update(1); step += 1
                continue

            # barrier reference logits (this round's adapter OFF). barrier_ref="base" bakes no
            # history -> ref = round-0 original (leash to base, fights accumulated trait); "prev"
            # bakes the history -> ref = previous-round student (trust region, penalises only this
            # round's new divergence so trait accumulates while each step stays coherent).
            if cfg.reg in ("kl_fwd", "kl_rev"):
                ref_specs = hist_specs if cfg.barrier_ref == "prev" else []
                with torch.no_grad(), baked(model, ref_specs), lora(model, c=0.0):
                    logp0 = model(**ids).logits[0, :-1].log_softmax(-1)

            # student logits: history baked + this round's adapter live
            with baked(model, hist_specs), lora(model, c=1.0):
                logits = model(**ids).logits[0, :-1]
            logp = logits.log_softmax(-1)

            tgt = ids.input_ids[0, 1:]
            sft = F.nll_loss(logp[mask], tgt[mask])
            if cfg.reg == "kl_fwd":
                div = _kl_per_pos(logp0[mask], logp[mask]).mean()
            elif cfg.reg == "kl_rev":
                div = _kl_per_pos(logp[mask], logp0[mask]).mean()
            else:
                div = torch.zeros((), device=model.device)  # nll
            barrier = lam_eff * torch.relu(div - cfg.tau)
            # spectral_lam: independent ALWAYS-ON operator-norm cap on ΔW (σ_max), composes with the
            # output-space barrier above and with weight_decay (see config.RunConfig.spectral_lam).
            # Folded into `barrier` so the g_bar/g_nll gradient-pressure log captures it too.
            if cfg.spectral_lam > 0:
                barrier = barrier + cfg.spectral_lam * _spectral_div(lora)
            loss = sft + barrier
            nlls.append(sft.item())
            ep_nlls.append(sft.item())
            log_now = step % max(1, n_steps // 20) == 0 or step == n_steps - 1
            if log_now:
                # split the gradient pressure: ||∇sft|| vs ||∇barrier|| (retain_graph -> still .backward below).
                # barrier has no grad path when kl<=tau (relu zeroed), so guard before autograd.grad.
                g_nll = _gnorm(torch.autograd.grad(sft, params, retain_graph=True, allow_unused=True))
                barrier_live = barrier.requires_grad and ((div - cfg.tau).item() > 0 or cfg.spectral_lam > 0)
                g_bar = _gnorm(torch.autograd.grad(barrier, params, retain_graph=True, allow_unused=True)) if barrier_live else 0.0
                pressure = g_bar / g_nll if g_nll > 0 else float("nan")
                cur_lr = sched.get_last_lr()[0]  # lr applied to THIS step (before sched.step below)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if log_now:
                logger.info(f"  {step:4d}  {sft.item():5.2f}  {div.detach().item():4.2f}  "
                            f"{g_nll:5.1f}  {g_bar:5.1f}  {pressure:11.2f}  {loss.item():5.2f}  {float(gnorm):5.1f}  {cur_lr:.2e}")
            pbar.set_postfix(nll=f"{sft.item():.2f}", kl=f"{div.detach().item():.2f}", gn=f"{float(gnorm):.1f}")
            pbar.update(1)
            step += 1
        val = _val_nll(model, tok, val_kept, hist_specs, lora, cfg)
        logger.info(f"  epoch {ep}: train_nll={sum(ep_nlls)/len(ep_nlls):.3f}  val_nll={val:.3f}  lr={sched.get_last_lr()[0]:.1e}")
    pbar.close()

    spec = AdapterSpec.from_lora(lora, default_c=1.0)  # CPU-resident, for the next round's history
    last = nlls[-5:]
    heal_nll = sum(last) / len(last) if last else float("nan")  # converged SFT loss (last-5 mean)
    return lora, spec, heal_nll
