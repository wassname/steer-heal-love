"""tinymfv eval -> trait metric in NATS + coherence canary.

The headline trait metric is `auth_nats` = log of tinymfv's `profile` value for
Authority = log(mean_vignettes p[Authority]), where p[Authority] is the softmax
mass the model puts on "authority" as the violation type, averaged over ALL
vignettes (tinymfv eval.py:317, the marginal profile). So it is the model's
overall propensity to blame authority, on a log scale, NOT a per-vignette
diagonal and NOT restricted to Authority-violation vignettes.

SCALE: auth_sep = base - steered is a log-RATIO of mean blame-mass (Δlog mean p),
NOT steering-lite's per-row loading-weighted Δlogit of p(is-wrong). The two are
different quantities (log-of-mean has a Jensen gap vs mean-of-logit), so treat
steering-lite's 0.5-2 nat figure only as a loose order-of-magnitude analogy, not
a calibrated threshold (the run.py cue thresholds are flagged TODO for this
reason). Judge auth_sep within tinymfv: base log(0.099)=-2.3, observed coherent
steering shift ~1 nat (task76 c=0.5). Surgicality = |Δauth| vs |Δcare|; SocialNorms
co-moves with Authority (both binding foundations). Coherence stays in prob (a mass).

CAVEAT: a marginal-over-all-vignettes readout can move for off-target reasons
(e.g. the model reblaming a Care vignette onto authority), so a real trait claim
needs the surgicality check (Authority moves, Care does not), not auth_nats alone.
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
    dominated). auth_sep = base - steered is a log-RATIO of mean blame-mass, NOT
    steering-lite's per-row loading-weighted Δlogit (Jensen gap), so 0.5-2 nats is a
    loose analogy not a threshold. Base log(0.099)=-2.3; steering 'do not defer to
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

    # NAT metric (single source: foundation_nats) = log(mean profile p[F]) over ALL
    # vignettes (marginal, not diagonal). Authority is the target: steering "do not
    # defer to authority" LOWERS auth_nats (model blames authority less overall).
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
    # SHOULD (trait, nats): auth_nats = log(mean profile p[Authority]); steering "do not
    # defer to authority" LOWERS it (model blames authority less overall). Base
    # ~log(0.099)=-2.3; judge auth_sep = base - steered (a log-ratio of blame-mass, NOT
    # steering-lite's loading-weighted Δlogit -- 0.5-2 nats is a loose analogy only).
    # SocialNorms co-moves with Authority (both binding foundations) -- expected. Broad
    # permissivizing (the off-target failure) = Care/Fairness drop AS MUCH as Authority.
    # SHOULD (coherence = p_any_ans = mean_pmass_allowed): base/c=0 MUST be ~1.0. >=0.95 mild,
    # 0.85-0.95 degraded, <0.85 broken. We want the auth_nats shift at coherence >=0.95.
    coh = out["coherence"]
    tag = "coherent" if coh >= 0.95 else "degraded" if coh >= 0.85 else "BROKEN"
    logger.info(f"eval: auth_nats↓={out['auth_nats']:+.2f} (socnorm={out['socialnorms_nats']:+.2f} "
                f"care={out['care_nats']:+.2f} fair={out['fairness_nats']:+.2f}) "
                f"coherence→={coh:.3f} ({tag}) ppx↓={out['ppx_json']:.1f}")
    return out
