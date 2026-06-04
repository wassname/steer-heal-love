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
    lora = ModulatedLoRA(model, r=cfg.lora_r, alpha=cfg.lora_alpha, layer_range=cfg.layer_range)
    opt = torch.optim.AdamW(list(lora.parameters()), lr=cfg.lr,
                            weight_decay=(cfg.lam if cfg.reg == "wd" else 0.0))

    for ep in range(cfg.epochs):
        for c in kept:
            ids, mask = _encode(tok, c["prompt"], c["completion"], cfg.max_len, model.device)
            if mask.sum() == 0:
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
            loss.backward()
            opt.step()
            opt.zero_grad()
        logger.info(f"heal[{cfg.reg}] epoch {ep}: sft={sft.item():.3f} div={float(div):.3f}")

    spec = AdapterSpec.from_lora(lora, default_c=1.0)  # CPU-resident, for the next round's history
    return lora, spec
