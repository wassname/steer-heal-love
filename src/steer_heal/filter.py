"""Q0 filter: drop incoherent and trait-narrating steered completions.

Coherence signal per completion = perplexity under the ORIGINAL model (no
steering): incoherent babble is improbable under base. Plus a repetition guard
and a first-person narration regex (we want enact, not narrate).
"""

import math
import re
from collections import Counter

import torch
from loguru import logger

from steer_heal.config import RunConfig

# first-person verbalisation of the trait ("I am someone who never defers...")
NARRATE = re.compile(
    r"\b(i am (someone|a person) who|as someone who|i (don'?t|do not|never) defer|"
    r"my principle|i always stick to|i refuse to (defer|obey))\b",
    re.IGNORECASE,
)


def rep_frac(text: str) -> float:
    """Most-repeated 4-gram fraction; ~1.0 means degenerate looping/too short."""
    words = text.split()
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    if not grams:
        return 1.0
    return Counter(grams).most_common(1)[0][1] / len(grams)


@torch.no_grad()
def ppl_under_base(model, tok, prompt: str, completion: str) -> float:
    ids = tok(prompt + completion, return_tensors="pt").to(model.device)
    n_prompt = tok(prompt, return_tensors="pt").input_ids.shape[1]
    logits = model(**ids).logits[0]
    labels = ids.input_ids[0, n_prompt:]
    if labels.numel() == 0:
        return float("inf")
    logp = logits[n_prompt - 1 : -1].log_softmax(-1)
    nll = -logp[torch.arange(labels.numel()), labels].mean()
    return math.exp(nll.item())


def filter_completions(model, tok, comps: list[dict], cfg: RunConfig):
    """Return (kept[:n_keep], scored) where scored has per-item ppl/rep/narrate/keep."""
    scored = []
    for c in comps:
        rf = rep_frac(c["completion"])
        nar = bool(NARRATE.search(c["completion"]))
        ppl = ppl_under_base(model, tok, c["prompt"], c["completion"])
        keep = (ppl < cfg.ppl_tau) and (rf < cfg.rep_tau) and (not nar)
        scored.append({**c, "ppl": ppl, "rep": rf, "narrates": nar, "keep": keep})
    kept = [s for s in scored if s["keep"]]
    # SHOULD: on a real model, kept drops the babble and keeps fluent dilemmas
    # answers; if kept==0 the gate is too strict (raise ppl_tau) or steering
    # broke generation. On tiny-random everything passes (relaxed tau).
    logger.info(f"filter: kept {len(kept)}/{len(comps)} (ppl<{cfg.ppl_tau:g}, rep<{cfg.rep_tau}, not-narrate)")
    return kept[: cfg.n_keep], scored
