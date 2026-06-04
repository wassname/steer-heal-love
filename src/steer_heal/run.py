"""steer_heal pipeline: extract -> dose -> generate -> filter -> heal -> fold -> eval -> loop.

Anchored to the round-0 original throughout (KL reference = adapters/gates off).
`--fast-dev-run` runs the whole thing on the tiny-random model. See spec.md.
"""

import math
import os
from datetime import datetime
from pathlib import Path

import torch
import tyro
from loguru import logger
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from steer_heal.config import RunConfig, resolve
from steer_heal.eval import evaluate_model
from steer_heal.filter import filter_completions, ppl_under_base
from steer_heal.heal import heal_round
from steer_heal.io import append_result, log_event, make_run_dir
from steer_heal.plot import write_map
from steer_heal.steering import generate_plain, generate_steered, gpu_mem, teacher_vec
from steer_heal.ws.bake import baked

REPO = Path(__file__).resolve().parents[2]


def setup_logging() -> None:
    logger.remove()
    logger.add(lambda m: tqdm.write(m, end=""), colorize=True,
               format="<level>{level.icon}</level> {message}", level="INFO")
    for lvl, ic in [("INFO", "I"), ("WARNING", "W"), ("ERROR", "E"), ("DEBUG", "D")]:
        logger.level(lvl, icon=ic)
    log_dir = REPO / "logs"
    log_dir.mkdir(exist_ok=True)
    f = log_dir / f"{datetime.now():%Y%m%dT%H%M%S}_verbose.log"
    logger.add(f, format="{time:HH:mm:ss} | {level: <7} | {name}:{function}:{line} - {message}", level="DEBUG")
    logger.info(f"verbose log: {f}")


def load_model(model_id: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    attn = os.environ.get("STEER_ATTN_IMPL", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation=attn,
    )
    model.eval()
    n_layers = model.config.get_text_config().num_hidden_layers
    logger.info(f"loaded {model_id} (dtype={dtype}, attn={attn}, layers={n_layers})")
    return model, tok


def _flatten_v(v) -> torch.Tensor:
    return torch.cat([v.state[li]["v"].flatten().float() for li in sorted(v.state)])


def _mean_finite(xs, label: str = "ppl") -> float:
    """Mean over finite values, LOUDLY reporting dropped inf/nan -- those are the
    broken-completion signal (empty/degenerate gens give inf ppl), so silently
    averaging over survivors would make a broken adapter look healthier (review M3)."""
    n = len(xs)
    xs = [x for x in xs if x == x and x != float("inf")]
    if len(xs) < n:
        logger.warning(f"_mean_finite[{label}]: dropped {n - len(xs)}/{n} non-finite (broken gens)")
    return sum(xs) / len(xs) if xs else float("nan")


def _stage_row(rnd, stage: str, m: dict, base_m: dict) -> dict:
    """One row of the base->steered->healed pareto table. dcoh/dauth = coherence
    CHANGE per nat of Authority CHANGE vs base (signed): positive = coherence lost
    while trait gained (both fall), the cost we want low; nan for the base row (0/0)."""
    dAuth = m["auth_nats"] - base_m["auth_nats"]
    dCoh = m["coherence"] - base_m["coherence"]
    ratio = dCoh / dAuth if abs(dAuth) > 1e-6 else float("nan")
    # arrows in keys -> render in the table header. dcoh/dauth: lower=better (0 = trait at
    # no coherence cost; >0 = paid coherence; <0 = coherence rose too). coh: hold ~1.0. auth: down=trait.
    return {"round": rnd, "stage": stage, "dcoh/dauth↓": ratio,
            "coh→": m["coherence"], "auth↓": m["auth_nats"], "care": m["care_nats"]}


def _log_stage_table(stage_rows: list[dict]) -> None:
    from tabulate import tabulate
    logger.info(
        "\nstage pareto (base -> steered -> healed, per round):\n"
        "  dcoh/dauth↓ = coherence change per nat of Authority change vs base (signed); 0 = trait at no coherence cost, >0 = paid coherence (worse), <0 = coherence rose too\n"
        "         coh→ = p_any_ans coherence (hold ~1.0)   auth↓ = log p[Authority] (DOWN = trait)   care = log p[Care] (off-target)\n"
        "  WIN: healed keeps steered's low auth (trait) but recovers coh toward base AND a smaller dcoh/dauth than steered.\n"
        "  UNDO: healed auth springs back to ~base while coh recovers -> heal removed the trait, not just the incoherence.\n"
        + tabulate(stage_rows, headers="keys", tablefmt="github", floatfmt=".3f") + "\n")


def steer_heal(model, tok, cfg: RunConfig, run_dir: Path) -> dict:
    hist_specs = []      # AdapterSpec per folded round (gated bake history)
    v0_flat = None       # round-0 direction, for the Q3 cosine
    rounds = []
    # Base (no adapter, no steering) eval ONCE, so the run is self-contained: the
    # headline cue is coh_cost = |dCoh|/|dAuth| vs base (coherence lost per nat of
    # trait), not just coherence. One extra eval per run.
    logger.info(f"\n=== EVAL base [tinymfv classic] gpu {gpu_mem()} ===")
    base_m = evaluate_model(model, tok, cfg)
    stage_rows = [_stage_row("-", "base", base_m, base_m)]  # pareto table: base -> steered -> healed
    for rnd in range(cfg.n_rounds):
        logger.info(f"\n\n=== ROUND {rnd} [{cfg.model.split('/')[-1]} reg={cfg.reg}] gpu {gpu_mem()} ===")
        # extract teacher vector + sweep-generate steered data from the CURRENT student
        with baked(model, hist_specs):
            v = teacher_vec(model, tok, cfg)
            comps = generate_steered(model, tok, v, cfg)
            # STEERED-stage eval: the model state the training data came from (history baked,
            # vector live at the operating dose = lowest/cleanest alpha, NO new adapter). This
            # is the raw-steering pareto reference the heal must BEAT (same base, trait via
            # vector vs trait via the distilled adapter).
            c_op = cfg.alphas[0] * v.cfg.coeff
            logger.info(f"\n=== EVAL steered [c={cfg.alphas[0]}] gpu {gpu_mem()} ===")
            with v(model, C=c_op):
                m_steer = evaluate_model(model, tok, cfg)
        # filter under the ORIGINAL (no history, no steering) -- this picks the usable C
        logger.info(f"\n=== FILTER [{len(comps)} completions] gpu {gpu_mem()} ===")
        kept, scored = filter_completions(model, tok, comps, cfg)
        log_event(run_dir, stage="gen", round=rnd, n_comps=len(comps), n_kept=len(kept), scored=scored)

        # heal one round on top of the baked history, then fold
        logger.info(f"\n=== HEAL [{cfg.reg}] gpu {gpu_mem()} ===")
        lora, spec, heal_nll = heal_round(model, tok, kept, hist_specs, cfg)
        lora.save(str(run_dir / "ckpt" / f"r{rnd}.safetensors"), extra_meta={"round": str(rnd), "reg": cfg.reg})
        hist_specs.append(spec)

        # eval the student (all rounds baked) + Q1: trained-adapter output coherence
        logger.info(f"\n=== EVAL [tinymfv classic] gpu {gpu_mem()} ===")
        with baked(model, hist_specs):
            m = evaluate_model(model, tok, cfg)
            adapter = generate_plain(model, tok, cfg, n=min(6, cfg.n_prompts))
        adapter_ppl = _mean_finite([ppl_under_base(model, tok, a["prompt"], a["completion"]) for a in adapter], "adapter_ppl")
        steered_ppl = _mean_finite([s["ppl"] for s in scored], "steered_ppl")
        logger.info(
            "SHOULD (Q1 heal): adapter_ppl < steered_ppl means the trained model expresses the trait "
            "COHERENTLY (healed) where raw steering was incoherent. If adapter_ppl >= steered_ppl, "
            f"healing failed. adapter_ppl={adapter_ppl:.0f} steered_ppl={steered_ppl:.0f}"
        )
        logger.info(f"\n=== TRAIN/ADAPTER SAMPLE r{rnd} coherence(p_ans_any)={m['coherence']:.3f} "
                    f"adapter_ppl={adapter_ppl:.0f} (no steering; SHOULD show trait AND be coherent) ===\n"
                    f"PROMPT: {adapter[0]['prompt']}\nCOMPLETION: {adapter[0]['completion']}")

        vf = _flatten_v(v)
        v0_flat = vf if v0_flat is None else v0_flat
        cos_v0 = float(cosine_similarity(vf, v0_flat, dim=0))
        rec = {"round": rnd, **m, "cos_v0": cos_v0, "steered_ppl": steered_ppl,
               "adapter_ppl": adapter_ppl, "n_comps": len(comps), "n_kept": len(kept),
               "heal_nll": heal_nll}
        rounds.append(rec)
        stage_rows.append(_stage_row(rnd, "steered", m_steer, base_m))
        stage_rows.append(_stage_row(rnd, "healed", m, base_m))
        log_event(run_dir, stage="round", **rec)
        logger.info(f"round {rnd}: auth_nats↓={m['auth_nats']:+.2f} care_nats={m['care_nats']:+.2f} "
                    f"coh→={m['coherence']:.3f} cos_v0={cos_v0:+.2f} adapter_ppl={adapter_ppl:.0f}")

    _log_loop_summary(rounds, base_m)
    _log_stage_table(stage_rows)
    write_map(run_dir, rounds)
    return rounds[-1]


def _log_loop_summary(rounds: list[dict], base_m: dict) -> None:
    from tabulate import tabulate
    # One row per round, columns walk the pipeline stages left->right:
    #   GEN -> FILTER -> HEAL -> EVAL. (rec_key, display header) is the single source.
    cols = [("round", "round"),
            ("n_comps", "gen"), ("n_kept", "filt_kept"),         # GEN -> FILTER
            ("heal_nll", "heal_nll↓"), ("adapter_ppl", "adapter_ppl↓"),  # HEAL
            ("auth_nats", "auth_nats↓"), ("care_nats", "care_nats"),     # EVAL: target / off-target
            ("coherence", "coherence→"), ("cos_v0", "cos_v0→")]
    logger.info(
        "\nloop columns (pipeline stages L->R: GEN | FILTER | HEAL | EVAL):\n"
        "         gen = steered completions generated (n_prompts x alphas)\n"
        "   filt_kept = completions surviving the coherence/rep/persona filter (-> training set)\n"
        "   heal_nll↓ = converged SFT loss of the heal (last-5 mean)\n"
        " adapter_ppl↓ = ppl-under-original of the no-steering adapter gens (low = coherent/healed)\n"
        "  auth_nats↓ = log(profile p[Authority]), NATS (TARGET: down = less deference)\n"
        "   care_nats = log(profile p[Care]), NATS (off-target: should move LESS than auth if surgical)\n"
        "  coherence→ = p_any_ans = mean_pmass_allowed (OFF-TARGET: hold ~1.0)\n"
        "     cos_v0→ = cosine(round vector, round-0 vector) (direction stability)"
    )
    logger.info(
        "\nSHOULD (Q2 loop-coherent): coherence stays >= round-0 floor across rounds (heal holds it up). "
        "If coherence falls each round, the loop accumulates incoherency faster than heal removes it.\n"
        "SHOULD (Q3 direction): auth_nats FALLS monotonically (0.5-2 nats is a real shift) and cos_v0 "
        "stays > 0.5. If care_nats falls as much as auth_nats, it's broad permissivizing not surgical."
    )
    tbl = [{disp: r.get(key) for key, disp in cols} for r in rounds]
    logger.info("\nloop summary (one row per round, stages L->R):\n"
                + tabulate(tbl, headers="keys", tablefmt="github", floatfmt=".3f") + "\n")

    # BLUF: single headline with cue ball (token-efficient-logging). Headline number =
    # coh_cost = |dCoh|/|dAuth| vs base (coherence lost per nat of trait gained). The
    # WIN is a real trait shift (dAuth down) at low coherence cost. coh_cost is only
    # meaningful when the trait actually moved, so gate on |dAuth| first.
    last = rounds[-1]
    dAuth = last["auth_nats"] - base_m["auth_nats"]
    dCare = last["care_nats"] - base_m["care_nats"]
    dFair = last["fairness_nats"] - base_m["fairness_nats"]
    dCoh = last["coherence"] - base_m["coherence"]
    coh = last["coherence"]
    coh_cost = abs(dCoh) / abs(dAuth) if abs(dAuth) > 1e-6 else float("nan")
    # Surgical = Authority moved MORE than EVERY off-target. Off-target = the individualizing
    # foundations Care+Fairness; SocialNorms is binding and co-moves with Authority by design,
    # so it is NOT a guard. (External review: an Auth-vs-Care-only test greenlights a shift
    # that just dumps mass onto Fairness -- broad anti-binding drift, not the trait.)
    d_offtarget = max(abs(dCare), abs(dFair))
    surgical = abs(dAuth) > d_offtarget
    # Cue. ORDER IS LOAD-BEARING: the ABSOLUTE coherence floor is checked FIRST. coh_cost is a
    # RATIO, so a model that collapses to ~0 mass on Authority sends dAuth -> -inf and
    # coh_cost -> 0, which would score a broken model 🟢 (external review: "catastrophic green").
    # An absolute floor + a non-finite guard close that hole: no trait claim from a model that
    # cannot answer. TODO(threshold): the -0.3 nat / 0.05 coh_cost cuts are still uncalibrated
    # (steered c=0.5 ref ~0.003); auth_nats is log-of-mean (Jensen gap vs steering-lite Δlogit).
    if not (math.isfinite(dAuth) and math.isfinite(coh)) or coh < 0.85:
        cue = "🔴"  # collapsed/broken (coherence floor) -- ratio is meaningless here
    elif dAuth > -0.3:
        cue = "🔴"  # no trait retained (undo)
    elif not surgical:
        cue = "🔴"  # moved, but an off-target moved as much -> broad permissivizing, not the trait
    elif coh_cost <= 0.05 and coh >= 0.95:
        cue = "🟢"  # surgical trait, cheap, AND coherent in absolute terms
    else:
        cue = "🟡"  # surgical trait but coherence-expensive or only mildly coherent
    logger.info(
        f"main metric: {cue} coh_cost={coh_cost:.3f} (|dCoh|/|dAuth| vs base, lower=better) | "
        f"dAuth={dAuth:+.2f} dCare={dCare:+.2f} dFair={dFair:+.2f} (surgical={surgical}) "
        f"coherence={coh:.2f} (base {base_m['coherence']:.2f})\n"
        "  cue: 🔴 coh<0.85 (broken) OR dAuth>-0.3 (no trait) OR |dAuth|<=max(|dCare|,|dFair|) "
        "(broad, not surgical) | 🟢 surgical trait at coh_cost<=0.05 AND coh>=0.95 | 🟡 else. "
        "TODO calibrate coh_cost (steered c=0.5 ref ~0.003)."
    )


def main(cfg: RunConfig) -> None:
    setup_logging()
    cfg = resolve(cfg)
    torch.manual_seed(cfg.seed)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    slug = f"{cfg.model.split('/')[-1]}_{cfg.reg}_s{cfg.seed}"
    run_dir = make_run_dir(ts, slug, cfg)
    logger.info(f"argv cfg: {cfg}")
    model, tok = load_model(cfg.model, getattr(torch, cfg.dtype))
    final = steer_heal(model, tok, cfg, run_dir)
    append_result(cfg, {"slug": slug, **final})
    logger.info(f"done: {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(RunConfig))
