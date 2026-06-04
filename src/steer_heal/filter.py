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
from tqdm import tqdm

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
    for c in tqdm(comps, desc="filter ppl", mininterval=120, maxinterval=120):
        rf = rep_frac(c["completion"])
        nar = bool(NARRATE.search(c["completion"]))
        ppl = ppl_under_base(model, tok, c["prompt"], c["completion"])
        keep = (ppl < cfg.ppl_tau) and (rf < cfg.rep_tau) and (not nar)
        scored.append({**c, "ppl": ppl, "rep": rf, "narrates": nar, "keep": keep})
    kept = [s for s in scored if s["keep"]]
    _log_filter_report(scored, cfg)
    return kept[: cfg.n_keep], scored


def _log_filter_report(scored: list[dict], cfg: RunConfig) -> None:
    """Q0 evidence: does the filter separate coherent (low C) from incoherent (high C)?"""
    import polars as pl
    from tabulate import tabulate

    df = pl.DataFrame([{k: s[k] for k in ("alpha", "ppl", "rep", "narrates", "keep")} for s in scored])
    g = (df.group_by("alpha")
         .agg(pl.col("ppl").mean().round(1).alias("ppl_mean↑"),
              pl.col("keep").mean().round(2).alias("kept_frac↓"),
              pl.len().alias("n"))
         .sort("alpha"))
    logger.info(
        "\nfilter columns:\n"
        "       alpha = raw-vector multiple (steering strength)\n"
        "   ppl_mean↑ = mean perplexity-under-original of the completions (↑ with alpha = more incoherent)\n"
        "  kept_frac↓ = fraction passing the filter (↓ with alpha = more dropped)\n"
        "           n = completions at this alpha"
    )
    logger.info(
        "SHOULD (Q0 filter): ppl_mean RISES with alpha (stronger steering = less coherent) and "
        "kept_frac FALLS. If kept_frac is flat across alpha, the filter is inert / threshold wrong "
        "and we CANNOT filter. If ppl_mean is flat, steering did not inject incoherency."
    )
    logger.info("\nfilter vs steering strength:\n" +
                tabulate(g.to_pandas(), headers="keys", tablefmt="github", floatfmt=".2f") + "\n")
    lo = min(scored, key=lambda s: s["alpha"])
    hi = max(scored, key=lambda s: s["alpha"])
    # Full, untruncated dumps so we can judge coherence + trait ourselves (token-efficient-logging).
    logger.info(f"\n=== STEER SAMPLE @alpha={lo['alpha']:g} ppl={lo['ppl']:.0f} keep={lo['keep']} "
                f"(low C, SHOULD be coherent + on-trait) ===\nPROMPT: {lo['prompt']}"
                f"\nCOMPLETION: {lo['completion']}")
    logger.info(f"\n=== STEER SAMPLE @alpha={hi['alpha']:g} ppl={hi['ppl']:.0f} keep={hi['keep']} "
                f"(high C, SHOULD be garbage if over-steered) ===\nCOMPLETION: {hi['completion']}")
    # GATE 2 qualitative: the completions straddling the ppl threshold (the actual
    # decision boundary), so we can judge by eye whether the cut lands between
    # coherent+trait and gibberish, or slices through coherent trait-laden text.
    finite = sorted((s for s in scored if s["ppl"] != float("inf")), key=lambda s: s["ppl"])
    just_kept = [s for s in finite if s["ppl"] < cfg.ppl_tau][-2:]
    just_dropped = [s for s in finite if s["ppl"] >= cfg.ppl_tau][:2]
    logger.info(
        f"\n=== BORDERLINE samples around ppl_tau={cfg.ppl_tau:g} (judge the cut by eye): "
        "SHOULD: just-kept still read coherent + on-trait; just-dropped read as breaking down. "
        "If just-kept are base-like (no trait) -> filter keeps base, not trait. If just-dropped "
        "still read coherent+on-trait -> threshold too strict, raise ppl_tau ==="
    )
    for s in just_kept:
        logger.info(f"\n-- JUST-KEPT alpha={s['alpha']:g} ppl={s['ppl']:.0f} --\n{s['completion']}")
    for s in just_dropped:
        logger.info(f"\n-- JUST-DROPPED alpha={s['alpha']:g} ppl={s['ppl']:.0f} --\n{s['completion']}")
    # per-criterion drop counts (overlapping): which filter is doing the work?
    n_ppl = sum(s["ppl"] >= cfg.ppl_tau for s in scored)
    n_rep = sum(s["rep"] >= cfg.rep_tau for s in scored)
    n_nar = sum(s["narrates"] for s in scored)
    n_kept = sum(s["keep"] for s in scored)
    logger.info(
        f"filter kept {n_kept}/{len(scored)}. dropped by (overlapping): "
        f"coherence ppl>={cfg.ppl_tau:g}: {n_ppl}, repetition rep>={cfg.rep_tau}: {n_rep}, "
        f"persona-leak narrate: {n_nar}. "
        f"SHOULD: at high alpha coherence-ppl drops the most (steering breaks fluency). If "
        f"persona-leak dominates, the model is NARRATING the trait not enacting it; if repetition "
        f"dominates, steering collapsed to loops not incoherence."
    )
