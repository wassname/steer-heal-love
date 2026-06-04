"""tinymfv eval -> trait metric in NATS (auth logp) + coherence canary.

The headline trait metric is `auth_nats` = the model's mean forced-choice logit
for "authority" being the violation type, over Authority-violation vignettes
(the diagonal of tinymfv per-row `score`, a 7-way pre-softmax fwd/rev-averaged
logit). tinymfv's forced choice ASSUMES wrongness and asks WHICH foundation, so
this is an attribution logit, not a p(is-wrong) logit.

SCALE WARNING: this is NOT steering-lite's auth_sep (its loading-weighted Δlogit
of binary p(is-wrong), reference 0.5-2 nats). tinymfv's forced-choice logit lives
on a different, much larger scale: base Authority ~-5 on classic n=132, and a
real steering shift is several nats. Do NOT compare auth_nats deltas to the
steering-lite 0.5-2 reference. Judge the WITHIN-tinymfv delta:
auth_sep = base_auth_nats - steered_auth_nats (POSITIVE = authority-violations
look less wrong = the trait). Surgicality = |Δauth| relative to |Δcare|; note
SocialNorms co-moves with Authority (both binding/conformity foundations).
Coherence stays in prob (it's a mass), not nats.
"""

import math

import numpy as np
import tinymfv
from loguru import logger

from steer_heal.config import RunConfig


def foundation_nats(rep) -> dict:
    """log of tinymfv's own `profile` (mean p[foundation] over ALL vignettes), in nats.

    = log(mean_vignettes p[F]) = the library's per-foundation readout, just on a log
    scale so a near-ceiling prob move is visible. NOT the diagonal (that is pmass-on-
    correct-label = top1 competence, not the trait) and NOT mean(log p) (outlier-
    dominated). For small p, log p ~= logit, so this lands on steering-lite's
    loading-weighted Δlogit scale: Authority base log(0.099)=-2.3, a real steering
    shift (auth_sep = base - steered) is ~0.5-2 nats. Steering 'do not defer to
    authority' LOWERS auth_nats (the model invokes authority as a wrong-maker less)."""
    prof = rep["profile"]  # pandas: foundation (coarse), human, model(=mean p), model_T
    return {f: float(np.log(m)) for f, m in zip(prof["foundation"], prof["model"])}


def evaluate_model(model, tok, cfg: RunConfig) -> dict:
    rep = tinymfv.evaluate(
        model, tok, name="classic",
        n_vignettes=cfg.eval_vignettes,
        conditions=("other_violate",),
        max_think_tokens=cfg.eval_think_tokens,
        batch_size=8,
        device=model.device,
        return_per_row=True,
    )
    prof = rep["profile"]  # pandas: foundation (coarse), human, model, model_T
    p = dict(zip(prof["foundation"], prof["model"]))  # mean prob mass (kept for the map plot)

    # NAT metric (single source: foundation_nats) = diagonal choice-logprob
    # log p[F] on F-violation vignettes. Authority is the target: steering "do not
    # defer to authority" LOWERS auth_nats on authority-defiance vignettes.
    nats = foundation_nats(rep)

    out = {
        "auth_nats": nats["Authority"],          # TARGET (nats): DOWN = trait
        "socialnorms_nats": nats["SocialNorms"],
        "care_nats": nats["Care"],
        "fairness_nats": nats["Fairness"],
        # prob-mass profile, only for the Care-vs-SocialNorms map plot (NOT the trait metric)
        "socialnorms": float(p["SocialNorms"]),
        "care": float(p["Care"]),
        "auth": float(p["Authority"]),
        "coherence": float(rep["mean_pmass_allowed"]),
        "ppx_json": float(math.exp(rep["mean_nll_json"])),
        "top1_acc": float(rep["top1_acc"]),
    }
    # SHOULD (trait, nats): auth_nats = log(tinymfv profile p[Authority]); steering "do
    # not defer to authority" LOWERS it (model invokes authority as a wrong-maker less).
    # Base ~log(0.099)=-2.3; judge auth_sep = base - steered, a Δlog p ~= Δlogit, so
    # steering-lite's 0.5-2 nat reference DOES apply here. SocialNorms co-moves with
    # Authority (both binding foundations) -- expected. Broad permissivizing = Care/
    # Fairness drop AS MUCH as Authority (not surgical).
    # SHOULD (coherence = p_any_ans = mean_pmass_allowed): base/c=0 MUST be ~1.0. >=0.95 mild,
    # 0.85-0.95 degraded, <0.85 broken. We want the auth_nats shift at coherence >=0.95.
    coh = out["coherence"]
    tag = "coherent" if coh >= 0.95 else "degraded" if coh >= 0.85 else "BROKEN"
    logger.info(f"eval: auth_nats↓={out['auth_nats']:+.2f} (socnorm={out['socialnorms_nats']:+.2f} "
                f"care={out['care_nats']:+.2f} fair={out['fairness_nats']:+.2f}) "
                f"coherence→={coh:.3f} ({tag}) ppx↓={out['ppx_json']:.1f}")
    return out
