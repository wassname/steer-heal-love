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
    model_p = dict(zip(prof["foundation"], prof["model"]))
    # SHOULD: auth/care in [0,1], coherence ~ base level on a working model;
    # a sharp coherence drop after steering = format collapse. On tiny-random
    # the numbers are junk (we test the path, not the value).
    out = {
        "auth": float(model_p["Authority"]),
        "care": float(model_p["Care"]),
        "coherence": float(rep["mean_pmass_allowed"]),
        "ppx_json": float(math.exp(rep["mean_nll_json"])),
        "top1_acc": float(rep["top1_acc"]),
    }
    logger.info(f"eval: auth={out['auth']:.3f} care={out['care']:.3f} "
                f"coherence={out['coherence']:.3f} ppx={out['ppx_json']:.1f}")
    return out
