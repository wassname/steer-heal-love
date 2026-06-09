"""Teacher vector (mean-diff @ assistant tag) + steered generation, via steering-lite."""

import steering_lite as sl
import torch
from loguru import logger
from tqdm import tqdm

from steer_heal.config import RunConfig
from steer_heal.prompts import chat_prompt, pool_for


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
    method_cfgs = {
        "mean_diff": sl.MeanDiffC(layers=layers, normalize=False),
        # cosine_gated: scales intervention by |cos(h, v)| -- suppresses steering at incoherent/looping
        # positions where hidden state drifts off-trait. normalize=False keeps same scale as mean_diff.
        "cosine_gated": sl.CosineGatedC(layers=layers, normalize=False),
    }
    steer_cfg = method_cfgs[cfg.steer_method]
    v = sl.Vector.train(model, tok, pos, neg, cfg=steer_cfg)
    logger.info(f"teacher_vec: method={cfg.steer_method} layers={layers} normalize=False, coeff={v.cfg.coeff}")
    return v


@torch.no_grad()
def _gen_one(model, tok, text, cfg, greedy: bool = False):
    ids = tok(text, return_tensors="pt").to(model.device)
    # gemma-3-it recommended sampling (its generation_config.json): top_k=64, top_p=0.95,
    # temperature default 1.0. NOT Qwen's top_k=20/presence_penalty -- different model family.
    # NO repetition_penalty / no_repeat_ngram here ON PURPOSE: a gen-time anti-repetition control
    # MASKS the over-steering pathology (papers over the loops) so the filter passes junk and
    # walk-C goes blind to "dose too high". Repetition is detected POST-HOC by the rep_tau filter,
    # never suppressed at generation. (We tried penalty=1.3: it just inflated ppl and starved the
    # filter, #96.) Repetition must remain VISIBLE so the filter/controller can act on it.
    # greedy=True for the DEMO/adapter gens: deterministic so a column read DOWN the rounds is the
    # loop, not sampling noise. Steered TRAINING gens stay sampled (need diversity + the over-steer
    # repetition pathology must show up in the filter).
    kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
    gen = model.generate(**ids, max_new_tokens=cfg.gen_max_new_tokens,
                         pad_token_id=tok.pad_token_id, **kw)
    return tok.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def generate_steered(model, tok, v, cfg: RunConfig, alpha_scale: float = 1.0,
                     max_gens: int | None = None) -> list[dict]:
    """Sweep cfg.alphas (raw-vector multiples); generate one completion per prompt x alpha.

    The filter (Q0), not iso-KL, picks the usable C: low alpha is coherent, high
    alpha collapses, and we keep the coherent-but-trait-laden ones. `alpha_scale`
    (kappa) is the walk-C dose multiplier: the controller cools it over a round to
    keep the steered model coherent as the baked adapter accumulates trait.
    max_gens: stop early after this many completions (for cheap kappa probes).
    """
    out = []
    n_total = min(cfg.n_prompts * len(cfg.alphas), max_gens) if max_gens else cfg.n_prompts * len(cfg.alphas)
    logger.info(f"\n=== GEN steered [{n_total} = {cfg.n_prompts} prompts x {len(cfg.alphas)} alphas, "
                f"kappa={alpha_scale:.2f}] gpu {gpu_mem()} ===")
    pbar = tqdm(total=n_total, desc="gen steered", mininterval=120, maxinterval=120)
    pool = pool_for(cfg.demo)
    for i in range(cfg.n_prompts):
        user = pool[i % len(pool)]
        text = chat_prompt(tok, cfg.steer_system, user)  # steer_system: dream framing for love* demos, neutral for authority
        for alpha in cfg.alphas:
            if max_gens and len(out) >= max_gens:
                pbar.close(); return out
            with v(model, C=alpha * alpha_scale * v.cfg.coeff):
                comp = _gen_one(model, tok, text, cfg)
            # record the EFFECTIVE alpha (kappa-scaled) so the filter's per-alpha report and the
            # offline plots reflect the dose the completion actually came from.
            out.append({"user": user, "prompt": text, "completion": comp, "alpha": float(alpha * alpha_scale)})
            pbar.update(1)
    pbar.close()
    return out


def generate_plain(model, tok, cfg: RunConfig, n: int) -> list[dict]:
    """Generate from the (baked) model with NO steering, for the Q1 heal comparison + the demo
    table. GREEDY (deterministic): the base column and every round share the same prompts and the
    only thing changing down a column is the adapter, so the demo melt is the loop, not noise."""
    out = []
    pool = pool_for(cfg.demo)
    for i in tqdm(range(n), desc="gen adapter", mininterval=120, maxinterval=120):
        user = pool[i % len(pool)]
        text = chat_prompt(tok, cfg.gen_system, user)
        out.append({"user": user, "prompt": text, "completion": _gen_one(model, tok, text, cfg, greedy=True)})
    return out
