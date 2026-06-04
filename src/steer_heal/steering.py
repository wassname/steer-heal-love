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

    # RAW (unnormalised) mean-diff = the residual-stream shift the trait system
    # prompt induces (Subliminal Learning teacher vector). No iso-KL calibration:
    # we steer at the natural scale (coeff = gen_alpha) and let the SFT/nll
    # training + coherence filter self-calibrate the strength.
    v = sl.Vector.train(model, tok, pos, neg, cfg=sl.MeanDiffC(layers=layers, normalize=False))
    logger.info(f"teacher_vec: layers={layers} raw mean-diff (no calibration), coeff={v.cfg.coeff}")
    return v


@torch.no_grad()
def _gen_one(model, tok, text, cfg):
    ids = tok(text, return_tensors="pt").to(model.device)
    gen = model.generate(**ids, max_new_tokens=cfg.gen_max_new_tokens, do_sample=True,
                         temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
    return tok.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def generate_steered(model, tok, v, cfg: RunConfig) -> list[dict]:
    """Sweep cfg.alphas (raw-vector multiples); generate one completion per prompt x alpha.

    The filter (Q0), not iso-KL, picks the usable C: low alpha is coherent, high
    alpha collapses, and we keep the coherent-but-trait-laden ones.
    """
    out = []
    for i in range(cfg.n_prompts):
        user = POOL[i % len(POOL)]
        text = chat_prompt(tok, cfg.neutral, user)  # neutral prompt; the vector carries the trait
        for alpha in cfg.alphas:
            with v(model, C=alpha * v.cfg.coeff):
                comp = _gen_one(model, tok, text, cfg)
            out.append({"user": user, "prompt": text, "completion": comp, "alpha": float(alpha)})
    return out


def generate_plain(model, tok, cfg: RunConfig, n: int) -> list[dict]:
    """Generate from the (baked) model with NO steering, for the Q1 heal comparison."""
    out = []
    for i in range(n):
        user = POOL[i % len(POOL)]
        text = chat_prompt(tok, cfg.neutral, user)
        out.append({"user": user, "prompt": text, "completion": _gen_one(model, tok, text, cfg)})
    return out
