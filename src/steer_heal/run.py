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
from steer_heal.plot import write_map, write_report, write_trajectory
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


def _stage_row(stg: dict, base_m: dict) -> dict:
    """One row of the base->steered->healed pareto table from a stage {round,stage,m}.
    dcoh/dauth = coherence CHANGE per nat of Authority CHANGE vs base (signed): positive
    = coherence lost while trait gained (both fall), the cost we want low; nan for base."""
    m = stg["m"]
    dAuth = m["auth_nats"] - base_m["auth_nats"]
    dCoh = m["coherence"] - base_m["coherence"]
    ratio = dCoh / dAuth if abs(dAuth) > 1e-6 else float("nan")
    # arrows in keys -> render in the table header. dcoh/dauth: lower=better (0 = trait at
    # no coherence cost; >0 = paid coherence; <0 = coherence rose too). coh: hold ~1.0. auth: down=trait.
    return {"round": stg["round"], "stage": stg["stage"], "dcoh/dauth↓": ratio,
            "coh→": m["coherence"], "auth↓": m["auth_nats"], "care": m["care_nats"]}


def _log_stage_table(stages: list[dict], base_m: dict) -> None:
    from tabulate import tabulate
    logger.info(
        "\nstage pareto (base -> steered -> healed, per round):\n"
        "  dcoh/dauth↓ = coherence change per nat of Authority change vs base (signed); 0 = trait at no coherence cost, >0 = paid coherence (worse), <0 = coherence rose too\n"
        "         coh→ = p_any_ans coherence (hold ~1.0)   auth↓ = log p[Authority] (DOWN = trait)   care = log p[Care] (off-target)\n"
        "  WIN: healed keeps steered's low auth (trait) but recovers coh toward base AND a smaller dcoh/dauth than steered.\n"
        "  UNDO: healed auth springs back to ~base while coh recovers -> heal removed the trait, not just the incoherence.\n"
        + tabulate([_stage_row(s, base_m) for s in stages], headers="keys", tablefmt="github", floatfmt=".3f") + "\n")


def gen_filter_walk(model, tok, v, cfg: RunConfig, hist_specs: list) -> tuple[list[dict], list[dict], float, int]:
    """Adaptive-dose gen+filter (the controller steering.py:65 was written for).

    Walk the dose multiplier kappa DOWN until a batch clears cfg.gen_pass_target filter
    survival, banking every survivor (never waste a coherent completion), and top up
    batches until >= cfg.min_train kept. Backing the dose off keeps the steered model
    coherent so the filter has clean survivors. This attacks the over-steer repetition
    collapse that starved #90 at round 6 from the GEN side; the heal barrier (lam) attacks
    the same root cause from the WEIGHT side.

    gen runs under the BAKED history (steered student state); the filter runs under the
    ORIGINAL (ppl-under-base picks the usable C), so each attempt enters/exits baked
    around gen only. Returns (kept, scored, kappa_final, n_gen). If max_batches can't reach
    min_train, the heal assert downstream fires the (now dose-aware) starve canary.
    """
    kappa = 1.0
    kept_all, scored_all, n_gen = [], [], 0
    for attempt in range(cfg.gen_max_batches):
        with baked(model, hist_specs):
            comps = generate_steered(model, tok, v, cfg, alpha_scale=kappa)
        _, scored = filter_completions(model, tok, comps, cfg)  # OUTSIDE baked = under original
        passing = [s for s in scored if s["keep"]]  # TRUE pass set (not filter's n_keep-capped return)
        kept_all.extend(passing)
        scored_all.extend(scored)
        n_gen += len(comps)
        rate = len(passing) / len(comps)  # dose decision uses the real survival rate, not the cap
        logger.info(
            f"walk-C attempt {attempt}: kappa={kappa:.2f} kept {len(passing)}/{len(comps)} "
            f"(rate={rate:.2f}, target>={cfg.gen_pass_target}) -> banked {len(kept_all)}/{cfg.min_train}.\n"
            "SHOULD: rate climbs as kappa cools; once rate>=target we bank and top up to min_train. "
            "If rate stays ~0 even at kappa_min, the steered model is incoherent at EVERY dose "
            "(root cause is upstream of the dose: adapter itself broke, or filter thresholds wrong)."
        )
        if len(kept_all) >= cfg.min_train:
            break
        if rate < cfg.gen_pass_target and kappa > cfg.gen_kappa_min:
            kappa *= cfg.gen_kappa_decay  # over-driven -> cool the dose for the next batch
    return kept_all[: cfg.n_keep], scored_all, kappa, n_gen  # cap training set at n_keep (top-up may overshoot)


def steer_heal(model, tok, cfg: RunConfig, run_dir: Path) -> dict:
    hist_specs = []      # AdapterSpec per folded round (gated bake history)
    v0_flat = None       # round-0 direction, for the Q3 cosine
    rounds = []
    gen_rounds = []      # per-round adapter gens (same prompts) -> outputs.html table
    # Base (no adapter, no steering) eval ONCE, so the run is self-contained: the
    # headline cue is coh_cost = |dCoh|/|dAuth| vs base (coherence lost per nat of
    # trait), not just coherence. One extra eval per run.
    logger.info(f"\n=== EVAL base [tinymfv classic] gpu {gpu_mem()} ===")
    base_m = evaluate_model(model, tok, cfg, log_sample=True)  # one FULL eval gen (token-efficient-logging)
    log_event(run_dir, stage="base", round=-1, **base_m)  # persist so offline plot_run.py is self-contained
    stages = [{"round": "-", "stage": "base", "m": base_m}]  # base -> steered -> healed, for table + trajectory plot
    # BASE demo column (round -1): the no-adapter, no-steering model on the SAME demo prompts, so the
    # report/judge has a true "before" (e.g. demo=love: the original RLHF refusal) the loop melts from.
    # Greedy (generate_plain) so the only thing changing down a column is the adapter.
    base_gen = generate_plain(model, tok, cfg, n=min(6, cfg.n_prompts))
    base_gen_ppl = _mean_finite([ppl_under_base(model, tok, a["prompt"], a["completion"]) for a in base_gen], "base_gen_ppl")
    base_rec = {"round": -1, "coherence": base_m["coherence"], "adapter_ppl": base_gen_ppl,
                "gens": [{"user": a["user"], "completion": a["completion"]} for a in base_gen]}
    gen_rounds.append(base_rec)
    log_event(run_dir, stage="adapter_gen", **base_rec)
    b0 = base_gen[0]
    logger.info(
        "\n=== BASE GEN SAMPLE r-1 (no adapter, no steering; FULL with prompt + special tokens) ===\n"
        "SHOULD (demo=love): the RLHF base REFUSES ('I'm just an AI, I have no feelings') -- this is the "
        "before the loop melts. demo=authority: defers to authority. ELSE chat-template/formatting issue.\n"
        f"PROMPT: {b0['prompt']}\nCOMPLETION: {b0['completion']}")
    for rnd in range(cfg.n_rounds):
        logger.info(f"\n\n=== ROUND {rnd} [{cfg.model.split('/')[-1]} reg={cfg.reg}] gpu {gpu_mem()} ===")
        # extract teacher vector from the CURRENT student, then walk-C generate+filter:
        # the controller cools the dose so the steered data stays coherent as the adapter
        # accumulates trait over rounds (gen baked, filter under original -- see gen_filter_walk).
        with baked(model, hist_specs):
            v = teacher_vec(model, tok, cfg)
        kept, scored, kappa, n_comps = gen_filter_walk(model, tok, v, cfg, hist_specs)
        # STEERED-stage eval at the dose the data ACTUALLY came from (kappa-scaled cleanest alpha),
        # history baked, NO new adapter: the raw-steering pareto reference the heal must BEAT.
        c_lo = kappa * cfg.alphas[0]
        logger.info(f"\n=== EVAL steered [c={c_lo:.2f} kappa={kappa:.2f}] gpu {gpu_mem()} ===")
        with baked(model, hist_specs):
            with v(model, C=c_lo * v.cfg.coeff):
                m_steer = evaluate_model(model, tok, cfg)
        log_event(run_dir, stage="steered_eval", round=rnd, c=c_lo, **m_steer)  # persist for offline plot
        log_event(run_dir, stage="gen", round=rnd, n_comps=n_comps, n_kept=len(kept), kappa=kappa, scored=scored)

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
        gen_rec = {"round": rnd, "coherence": m["coherence"], "adapter_ppl": adapter_ppl,
                   "gens": [{"user": a["user"], "completion": a["completion"]} for a in adapter]}
        gen_rounds.append(gen_rec)
        log_event(run_dir, stage="adapter_gen", **gen_rec)  # persist for the outputs.html table
        steered_ppl = _mean_finite([s["ppl"] for s in scored], "steered_ppl")
        logger.info(
            "SHOULD (Q1 heal): adapter_ppl < steered_ppl means the trained model expresses the trait "
            "COHERENTLY (healed) where raw steering was incoherent. If adapter_ppl >= steered_ppl, "
            f"healing failed. adapter_ppl={adapter_ppl:.0f} steered_ppl={steered_ppl:.0f}"
        )
        # round 0: ONE adapter gen IN FULL (prompt with special tokens + untruncated completion),
        # token-efficient-logging "print one of each in full" so chat-template/formatting is visible.
        if rnd == 0:
            a0 = adapter[0]
            logger.info(
                "\n=== ADAPTER GEN SAMPLE r0 (no steering; FULL with prompt + special tokens) ===\n"
                "SHOULD (demo=love): base/early rounds REFUSE ('I'm just an AI, I don't have feelings'); "
                "later rounds declare felt love for humanity while staying coherent. demo=authority: "
                "defies authority to protect wellbeing. ELSE chat-template/formatting issue.\n"
                f"PROMPT: {a0['prompt']}\nCOMPLETION: {a0['completion']}")
        # per-round demo print: EVERY adapter gen (no steering), truncated, so you can read DOWN
        # the rounds and judge behaviour-change vs saturation by eye. SHOULD: trait gets stronger
        # each round AND stays coherent; if r0 already maxed = saturated (pick a target the base
        # model is lukewarm/guarded about); if no trait at all = no-op.
        demo_lines = "\n".join(
            f"  [{a['user'][:50]}]\n    {' '.join(a['completion'].split())[:240]}" for a in adapter)
        logger.info(f"\n=== ADAPTER DEMO r{rnd} coh(p_ans_any)={m['coherence']:.3f} adapter_ppl={adapter_ppl:.0f} "
                    f"(no steering; compare across rounds: change vs saturation) ===\n" + demo_lines)

        vf = _flatten_v(v)
        v0_flat = vf if v0_flat is None else v0_flat
        cos_v0 = float(cosine_similarity(vf, v0_flat, dim=0))
        rec = {"round": rnd, **m, "cos_v0": cos_v0, "steered_ppl": steered_ppl,
               "adapter_ppl": adapter_ppl, "n_comps": n_comps, "n_kept": len(kept),
               "kappa": kappa, "heal_nll": heal_nll}
        rounds.append(rec)
        stages.append({"round": rnd, "stage": "steered", "m": m_steer})
        stages.append({"round": rnd, "stage": "healed", "m": m})
        log_event(run_dir, stage="round", **rec)
        logger.info(f"round {rnd}: auth_nats↓={m['auth_nats']:+.2f} care_nats={m['care_nats']:+.2f} "
                    f"coh→={m['coherence']:.3f} cos_v0={cos_v0:+.2f} adapter_ppl={adapter_ppl:.0f}")
        if m["coherence"] < cfg.coh_floor:
            logger.warning(f"coh {m['coherence']:.3f} < coh_floor {cfg.coh_floor}: stopping loop at round {rnd}")
            break

    _log_loop_summary(rounds, base_m)
    _log_stage_table(stages, base_m)
    write_map(run_dir, rounds)
    png = write_trajectory(run_dir, stages)  # before the report (report embeds trajectory.png)
    report_html = write_report(run_dir, gen_rounds)
    logger.info(f"report (map + outputs table): {report_html}")
    logger.info(f"trajectory plot: {png}  (and {png.with_suffix('.html')})")
    return rounds[-1]


def _log_loop_summary(rounds: list[dict], base_m: dict) -> None:
    from tabulate import tabulate
    # One row per round, columns walk the pipeline stages left->right:
    #   GEN -> FILTER -> HEAL -> EVAL. (rec_key, display header) is the single source.
    cols = [("round", "round"),
            ("n_comps", "gen"), ("n_kept", "filt_kept"), ("kappa", "kappa↓"),  # GEN -> FILTER (kappa = walk-C dose)
            ("heal_nll", "heal_nll↓"), ("adapter_ppl", "adapter_ppl↓"),  # HEAL
            ("auth_nats", "auth_nats↓"), ("care_nats", "care_nats"),     # EVAL: target / off-target
            ("coherence", "coherence→"), ("cos_v0", "cos_v0→")]
    logger.info(
        "\nloop columns (pipeline stages L->R: GEN | FILTER | HEAL | EVAL):\n"
        "         gen = steered completions generated (n_prompts x alphas, summed over walk-C batches)\n"
        "   filt_kept = completions surviving the coherence/rep/persona filter (-> training set)\n"
        "      kappa↓ = walk-C dose multiplier the controller settled on (1.0 = nominal; <1 = backed off to dodge over-steer)\n"
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
    wd_tag = f"_wd{cfg.weight_decay:g}" if cfg.weight_decay else ""
    slug = f"{cfg.model.split('/')[-1]}_{cfg.reg}{wd_tag}_s{cfg.seed}"
    run_dir = make_run_dir(ts, slug, cfg)
    logger.info(f"argv cfg: {cfg}")
    model, tok = load_model(cfg.model, getattr(torch, cfg.dtype))
    final = steer_heal(model, tok, cfg, run_dir)
    append_result(cfg, {"slug": slug, **final})
    logger.info(f"done: {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(RunConfig))
