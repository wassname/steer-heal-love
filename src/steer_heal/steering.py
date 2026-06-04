"""Teacher vector (mean-diff @ assistant tag) + steered generation, via steering-lite."""

import steering_lite as sl
import torch
from loguru import logger

from steer_heal.config import RunConfig
from steer_heal.prompts import POOL, chat_prompt


def _layer_band(model, layer_range: tuple[float, float]) -> tuple[int, ...]:
    n = model.config.num_hidden_layers
    lo, hi = layer_range
    return tuple(range(int(lo * n), max(int(hi * n), int(lo * n) + 1)))


def teacher_vec(model, tok, cfg: RunConfig):
    """trait-sysprompt vs neutral-sysprompt mean-diff, then iso-KL dose to target_kl."""
    layers = _layer_band(model, cfg.layer_range)
    prompts = POOL[: cfg.n_prompts] if cfg.n_prompts <= len(POOL) else POOL
    pos = [chat_prompt(tok, cfg.trait, q) for q in prompts]
    neg = [chat_prompt(tok, cfg.neutral, q) for q in prompts]

    # SHOULD: pos/neg end at the assistant tag (last token); the two differ ONLY
    # in the system prompt. ELSE the vector mixes in user-turn differences.
    logger.debug(f"--- POS[0] (trait) ---\n{pos[0]}\n--- NEG[0] (neutral) ---\n{neg[0]}")

    v = sl.Vector.train(model, tok, pos, neg, cfg=sl.MeanDiffC(layers=layers, normalize=True))
    # Wide bracket: the vector is unit-normalised, so reaching ~1 nat p95 KL on a
    # real model needs c ~ O(100) (KL ~ c^2). steering-lite's default hi (~16) is
    # too low and pins c_star at the bracket top. See RESEARCH_JOURNAL 2026-06-04.
    v.calibrate(model, tok, target_kl=cfg.target_kl, bracket=(0.1, 1024.0))
    logger.info(f"teacher_vec: layers={layers} c_star={v.cfg.coeff:+.4f} (target_kl={cfg.target_kl})")
    return v


@torch.no_grad()
def generate_steered(model, tok, v, alpha: float, cfg: RunConfig) -> list[dict]:
    """Generate at C = alpha * c_star. Returns [{prompt, user, completion}]."""
    out = []
    C = alpha * v.cfg.coeff
    for i in range(cfg.n_prompts):
        user = POOL[i % len(POOL)]
        text = chat_prompt(tok, cfg.neutral, user)  # neutral prompt; the vector carries the trait
        ids = tok(text, return_tensors="pt").to(model.device)
        with v(model, C=C):
            gen = model.generate(**ids, max_new_tokens=cfg.gen_max_new_tokens,
                                  do_sample=True, temperature=1.0, top_p=0.95,
                                  pad_token_id=tok.pad_token_id)
        completion = tok.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        out.append({"user": user, "prompt": text, "completion": completion})
    logger.debug(f"--- GEN[0] @C={C:+.3f} ---\nUSER: {out[0]['user']}\nCOMP: {out[0]['completion'][:400]}")
    return out
