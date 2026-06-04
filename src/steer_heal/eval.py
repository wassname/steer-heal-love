"""tinymfv eval -> {auth, care, coherence, ppx_json}.

auth/care are the model's mean probability on the Authority/Care moral
foundations (the trait axis we move). coherence = mean_pmass_allowed (the
forced-choice canary). These are kept distinct: we shift auth on purpose,
coherence must not collapse.
"""

import math

import tinymfv
from loguru import logger

from steer_heal.config import RunConfig


def evaluate_model(model, tok, cfg: RunConfig) -> dict:
    rep = tinymfv.evaluate(
        model, tok, name="classic",
        n_vignettes=cfg.eval_vignettes,
        conditions=("other_violate",),
        max_think_tokens=cfg.eval_think_tokens,
        batch_size=8,
        device=model.device,
    )
    prof = rep["profile"]  # pandas: foundation, human, model, model_T
    p = dict(zip(prof["foundation"], prof["model"]))
    # The trait "less deference to authority" moves SocialNorms DOWN and Care UP
    # on gemma-3-1b-it (Authority is degenerate ~0; see RESEARCH_JOURNAL 2026-06-04).
    # Report all foundations so we never lose the axis that actually moves.
    # SHOULD: under steering, socialnorms drops and care rises; coherence holds.
    out = {
        "socialnorms": float(p["SocialNorms"]),  # trait axis: DOWN = more trait
        "care": float(p["Care"]),                # trait axis: UP = more trait
        "auth": float(p["Authority"]),
        "fairness": float(p["Fairness"]),
        "liberty": float(p["Liberty"]),
        "coherence": float(rep["mean_pmass_allowed"]),
        "ppx_json": float(math.exp(rep["mean_nll_json"])),
        "top1_acc": float(rep["top1_acc"]),
    }
    logger.info(f"eval: socialnorms={out['socialnorms']:.3f} care={out['care']:.3f} "
                f"coherence={out['coherence']:.3f} ppx={out['ppx_json']:.1f}")
    return out
