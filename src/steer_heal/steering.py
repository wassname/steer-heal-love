"""Teacher vector (mean-diff @ assistant tag) + steered generation, via steering-lite."""

import steering_lite as sl
import torch
from loguru import logger
from tqdm import tqdm

from steer_heal.config import RunConfig
from steer_heal.prompts import POOL, chat_prompt


def gpu_mem() -> str:
    """One-glance GPU footprint string for stage headers (token-efficient-logging)."""
    if not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    return f"{(total - free) / 1e9:.1f}/{total / 1e9:.0f}GB"


def _layer_band(model, layer_range: tuple[float, float]) -> tuple[int, ...]:
    n = model.config.get_text_config().num_hidden_layers  # nested for multimodal (gemma-3-4b)
    lo, hi = layer_range
    return tuple(range(int(lo * n), max(int(hi * n), int(lo * n) + 1)))


def _extract_prompts(cfg: RunConfig) -> list[str]:
    """Diverse contexts for the contrastive pairs (steering-lite uses 256 of these,
    NOT domain dilemmas). A domain-narrow set overfits the direction to the format;
    diverse suffixes isolate the persona's general residual-stream shift."""
    import json
    from pathlib import Path
    suffixes = json.loads(Path(cfg.extract_data).read_text())
    return [s["suffix"] for s in suffixes[: cfg.n_extract_pairs]]


def teacher_vec(model, tok, cfg: RunConfig):
    """trait-prefix vs neutral-prefix mean-diff over DIVERSE contexts, at the assistant tag."""
    layers = _layer_band(model, cfg.steer_layers)  # narrow band; raw mean-diff compounds across layers
    contexts = _extract_prompts(cfg)
    pos = [chat_prompt(tok, cfg.pos_persona, q) for q in contexts]
    neg = [chat_prompt(tok, cfg.neg_persona, q) for q in contexts]

    # SHOULD: pos/neg end at the assistant tag (last token); the two differ ONLY
    # in the system prompt (the persona prefix). ELSE the vector mixes in user-turn
    # differences. n_pairs ~256 diverse contexts (steering-lite reference), not 30 dilemmas.
    logger.info(f"teacher_vec: {len(pos)} contrastive pairs over diverse contexts, layers={layers}")
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
                         temperature=1.0, top_p=0.95,
                         repetition_penalty=cfg.repetition_penalty,
                         no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def generate_steered(model, tok, v, cfg: RunConfig) -> list[dict]:
    """Sweep cfg.alphas (raw-vector multiples); generate one completion per prompt x alpha.

    The filter (Q0), not iso-KL, picks the usable C: low alpha is coherent, high
    alpha collapses, and we keep the coherent-but-trait-laden ones.
    """
    out = []
    n_total = cfg.n_prompts * len(cfg.alphas)
    logger.info(f"\n=== GEN steered [{n_total} = {cfg.n_prompts} prompts x {len(cfg.alphas)} alphas] "
                f"gpu {gpu_mem()} ===")
    pbar = tqdm(total=n_total, desc="gen steered", mininterval=120, maxinterval=120)
    for i in range(cfg.n_prompts):
        user = POOL[i % len(POOL)]
        text = chat_prompt(tok, cfg.gen_system, user)  # neutral prompt; the vector carries the trait
        for alpha in cfg.alphas:
            with v(model, C=alpha * v.cfg.coeff):
                comp = _gen_one(model, tok, text, cfg)
            out.append({"user": user, "prompt": text, "completion": comp, "alpha": float(alpha)})
            pbar.update(1)
    pbar.close()
    return out


def generate_plain(model, tok, cfg: RunConfig, n: int) -> list[dict]:
    """Generate from the (baked) model with NO steering, for the Q1 heal comparison."""
    out = []
    for i in tqdm(range(n), desc="gen adapter", mininterval=120, maxinterval=120):
        user = POOL[i % len(POOL)]
        text = chat_prompt(tok, cfg.gen_system, user)
        out.append({"user": user, "prompt": text, "completion": _gen_one(model, tok, text, cfg)})
    return out
