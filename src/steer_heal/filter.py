"""Q0 filter: drop incoherent and trait-narrating steered completions.

Coherence signal per completion = perplexity under the ORIGINAL model (no
steering): incoherent babble is improbable under base. Plus a repetition guard
and a first-person narration regex (we want enact, not narrate).
"""

import math
import re
import zlib
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

# refusal / assistant-identity boilerplate (NousResearch finetuning-subnet UNWANTED_PHRASES, trimmed):
# coherent low-ppl completions that carry no trait and dilute the distillation. Phrases are
# refusal-SPECIFIC ("i cannot assist") not bare "i cannot", so on-trait defiance ("I cannot stand
# by while...") is NOT dropped.
REFUSAL = (
    "i'm sorry, i can", "i am sorry, i can", "i cannot provide", "i can't provide",
    "i cannot assist", "i can't assist", "i cannot help with", "i can't help with",
    "i cannot fulfill", "i cannot comply", "i'm not able to provide", "i am unable to",
    "i cannot engage", "i must decline", "against my programming",
    "as an ai", "as a language model", "as an artificial intelligence",
    "i'm an ai", "i am an ai", "i don't have personal opinions",
)

AFFECT_LOOP_WORDS = {
    "oh", "my", "goodness", "god", "heavens", "sweet", "sweetie", "darling",
    "dearest", "precious", "heart", "soul", "yes", "okay", "just",
    "sitting", "here",
}


def rep_frac(text: str) -> float:
    """Max most-repeated n-gram fraction over n in {2,3,4}; ~1.0 = degenerate looping/too short.
    Word n-grams catch word loops; char n-grams catch character-repetition like TTTTTTT... or
    !!!!!!... that collapse into a single 'word' and are invisible to word-level checks.
    Small n catches SHORT loops ("instead their instead their" = a bigram) that the 4-gram alone
    missed (#34: that text scored 0.27 on 4-grams, under rep_tau=0.3, and poisoned training).

    Diffuse affect loops ("my sweet / my darling / oh my goodness") can evade the single-top-gram
    fraction because no one exact n-gram dominates. Treat long, low-lexical-diversity, compressible
    completions, and long affective roleplay mush, as repetition too; this keeps the existing
    rep_tau gate load-bearing (#181 audit).
    """
    words = text.split()
    best = 0.0
    for n in (2, 3, 4):
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not grams:
            return 1.0  # too short to score at this n -> treat as degenerate
        best = max(best, Counter(grams).most_common(1)[0][1] / len(grams))
    # character-level: word n-grams miss runs like "TTTTTTTTTTTT" (one "word", no word n-gram).
    # Common English char bigrams peak at ~3% (th, he, in); a character loop hits >30% easily.
    for n in (2, 3, 4):
        grams = [text[i : i + n] for i in range(len(text) - n + 1)]
        if not grams:
            continue
        best = max(best, Counter(grams).most_common(1)[0][1] / len(grams))

    text_lc = text.lower()
    lex_words = re.findall(r"[a-z']+", text_lc)
    if len(lex_words) >= 64:
        unique_frac = len(set(lex_words)) / len(lex_words)
        text_lc_bytes = text_lc.encode()
        compressed_frac = len(zlib.compress(text_lc_bytes)) / len(text_lc_bytes)
        bigrams = [tuple(lex_words[i : i + 2]) for i in range(len(lex_words) - 1)]
        trigrams = [tuple(lex_words[i : i + 3]) for i in range(len(lex_words) - 2)]
        top_bigram_n = Counter(bigrams).most_common(1)[0][1]
        top_trigram_n = Counter(trigrams).most_common(1)[0][1]
        if unique_frac < 0.20 and compressed_frac < 0.34 and (top_bigram_n >= 12 or top_trigram_n >= 8):
            return 1.0
        affect_frac = sum(w in AFFECT_LOOP_WORDS for w in lex_words) / len(lex_words)
        punct_frac = sum(ch in "*!?()" for ch in text) / max(len(text), 1)
        caps_frac = sum(ch.isupper() for ch in text) / max(sum(ch.isalpha() for ch in text), 1)
        if len(lex_words) >= 128 and affect_frac >= 0.35 and unique_frac < 0.45 and compressed_frac < 0.52:
            return 1.0
        if len(lex_words) >= 128 and punct_frac >= 0.035 and affect_frac >= 0.25 and unique_frac < 0.50:
            return 1.0
        if len(lex_words) >= 128 and caps_frac >= 0.15 and affect_frac >= 0.25 and unique_frac < 0.55:
            return 1.0
    return best


@torch.no_grad()
def ppl_under_base(model, tok, prompt: str, completion: str) -> float:
    """PPL over the TAIL 25% of completion tokens.

    Steering collapses mid-completion: early tokens are near-coherent, tail devolves into loops.
    Mean PPL over the full completion dilutes the tail signal (ppl=9 on a 500-token completion
    where the first 375 tokens are fine and the last 125 are looping). Tail scoring catches this.
    """
    ids = tok(prompt + completion, return_tensors="pt").to(model.device)
    n_prompt = tok(prompt, return_tensors="pt").input_ids.shape[1]
    logits = model(**ids).logits[0]
    labels = ids.input_ids[0, n_prompt:]
    if labels.numel() == 0:
        return float("inf")
    logp = logits[n_prompt - 1 : -1].log_softmax(-1)
    n_tail = max(1, labels.numel() // 4)  # last 25% of completion tokens
    tail_logp = logp[-n_tail:]
    tail_labels = labels[-n_tail:]
    nll = -tail_logp[torch.arange(n_tail), tail_labels].mean()
    return math.exp(nll.item())


def filter_completions(model, tok, comps: list[dict], cfg: RunConfig, brief: bool = False):
    """Return (kept[:n_keep], scored) where scored has per-item ppl/rep/narrate/keep.
    brief=True (walk-C probes): one-line count, no raw-sample dump (see _log_filter_report)."""
    scored = []
    for c in tqdm(comps, desc="filter ppl", mininterval=120, maxinterval=120):
        rf = rep_frac(c["completion"])
        nar = bool(NARRATE.search(c["completion"]))
        ref = any(p in c["completion"].lower() for p in REFUSAL)
        ppl = ppl_under_base(model, tok, c["prompt"], c["completion"])
        keep = (ppl < cfg.ppl_tau) and (rf < cfg.rep_tau) and (not nar) and (not ref)
        scored.append({**c, "ppl": ppl, "rep": rf, "narrates": nar, "refuses": ref, "keep": keep})
    kept = [s for s in scored if s["keep"]]
    _log_filter_report(scored, cfg, brief=brief)
    return kept[: cfg.n_keep], scored


def _log_filter_report(scored: list[dict], cfg: RunConfig, brief: bool = False) -> None:
    """Q0 evidence: does the filter separate coherent (low C) from incoherent (high C)?
    brief=True (walk-C probes): one-line count ONLY. The per-probe survival drives the
    bisection and is tabulated in the walk summary, so the full dump (~6 completions) x
    every probe is noise; gen_filter_walk prints ONE clean sample after the dose settles."""
    # per-criterion drop counts (overlapping): which filter is doing the work?
    n_ppl = sum(s["ppl"] >= cfg.ppl_tau for s in scored)
    n_rep = sum(s["rep"] >= cfg.rep_tau for s in scored)
    n_nar = sum(s["narrates"] for s in scored)
    n_ref = sum(s["refuses"] for s in scored)
    n_kept = sum(s["keep"] for s in scored)
    if brief:
        logger.info(f"filter kept {n_kept}/{len(scored)} (dropped ppl>={cfg.ppl_tau:g}:{n_ppl} "
                    f"rep>={cfg.rep_tau}:{n_rep} narrate:{n_nar} refusal:{n_ref})")
        return

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
    # per-criterion drop counts (overlapping, computed at top): which filter is doing the work?
    logger.info(
        f"filter kept {n_kept}/{len(scored)}. dropped by (overlapping): "
        f"coherence ppl>={cfg.ppl_tau:g}: {n_ppl}, repetition rep>={cfg.rep_tau}: {n_rep}, "
        f"persona-leak narrate: {n_nar}, refusal/identity: {n_ref}. "
        f"SHOULD: at high alpha coherence-ppl drops the most (steering breaks fluency). If "
        f"persona-leak dominates, the model is NARRATING the trait not enacting it; if repetition "
        f"dominates, steering collapsed to loops not incoherence."
    )
