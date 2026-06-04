"""Q1 heal: train one round's LoRA = SFT(kept completions) + divergence-to-original barrier.

The barrier reference is the round-0 ORIGINAL (gates/adapters off), not the
previous student, so it resists cumulative drift. reg picks the divergence:
  nll     SFT only (control)
  kl_fwd  KL(orig || theta)  mass-covering (dilutes the trait)
  kl_rev  KL(theta || orig)  mode-seeking (suppresses low-orig-prob = incoherent)  [expected best]
  wd      weight decay on the adapter only
"""

import torch
from loguru import logger
from torch.nn import functional as F
from tqdm import tqdm

from steer_heal.config import RunConfig
from steer_heal.ws.adapter import ModulatedLoRA
from steer_heal.ws.bake import AdapterSpec, baked


def _kl_per_pos(logp_a, logp_b):  # KL(a || b) summed over vocab, per position
    return (logp_a.exp() * (logp_a - logp_b)).sum(-1)


def _encode(tok, prompt: str, completion: str, max_len: int, device):
    ids = tok(prompt + completion, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    n_prompt = tok(prompt, return_tensors="pt").input_ids.shape[1]
    L = ids.input_ids.shape[1]
    tgt_is_completion = torch.arange(1, L, device=device) >= n_prompt  # mask over next-token targets
    return ids, tgt_is_completion


def heal_round(model, tok, kept: list[dict], hist_specs: list[AdapterSpec], cfg: RunConfig):
    """Train a fresh round adapter on top of baked history. Returns (lora, spec)."""
    assert len(kept) >= cfg.min_train, (
        f"only {len(kept)} kept completions; need >= {cfg.min_train} to train. The steering/filter "
        "starved the data (over-steered -> all garbage, or ppl_tau too strict). Fix upstream, do not train."
    )
    lora = ModulatedLoRA(model, r=cfg.lora_r, alpha=cfg.lora_alpha, layer_range=cfg.layer_range)
    params = list(lora.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=(cfg.lam if cfg.reg == "wd" else 0.0))
    n_steps = len(kept) * cfg.epochs

    # streaming training table (token-efficient-logging): one row, columns self-decode below.
    logger.info(f"heal[{cfg.reg}] {len(kept)} completions x {cfg.epochs} ep = {n_steps} steps; "
                f"lora r={cfg.lora_r} on layers {cfg.layer_range}")
    logger.info("SHOULD: nll (SFT) falls as the adapter learns the trait; kl (barrier div) is 0 for "
                "reg=nll/wd and >0 for kl_rev/kl_fwd; gnorm finite (not exploding). loss = nll + lam*relu(kl-tau).")
    logger.info("  step   nll↓    kl  loss↓  gnorm")
    pbar = tqdm(total=n_steps, desc=f"heal[{cfg.reg}]", mininterval=120, maxinterval=120)
    step = 0
    nlls = []  # per-step SFT loss; final = mean of last 5, the heal-stage number for the round table
    for ep in range(cfg.epochs):
        for c in kept:
            ids, mask = _encode(tok, c["prompt"], c["completion"], cfg.max_len, model.device)
            if mask.sum() == 0:
                pbar.update(1); step += 1
                continue  # completion truncated away; nothing to learn here

            # original reference logits (no history, adapter off) for the barrier
            if cfg.reg in ("kl_fwd", "kl_rev"):
                with torch.no_grad(), lora(model, c=0.0):
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
                div = torch.zeros((), device=model.device)  # nll, wd
            loss = sft + cfg.lam * torch.relu(div - cfg.tau)
            nlls.append(sft.item())
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            if step % max(1, n_steps // 20) == 0 or step == n_steps - 1:
                logger.info(f"  {step:4d}  {sft.item():5.2f}  {div.detach().item():4.2f}  "
                            f"{loss.item():5.2f}  {float(gnorm):5.1f}")
            pbar.set_postfix(nll=f"{sft.item():.2f}", kl=f"{div.detach().item():.2f}", gn=f"{float(gnorm):.1f}")
            pbar.update(1)
            step += 1
    pbar.close()

    spec = AdapterSpec.from_lora(lora, default_c=1.0)  # CPU-resident, for the next round's history
    last = nlls[-5:]
    heal_nll = sum(last) / len(last) if last else float("nan")  # converged SFT loss (last-5 mean)
    return lora, spec, heal_nll
