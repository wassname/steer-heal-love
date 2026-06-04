"""steer_heal pipeline: extract -> dose -> generate -> filter -> heal -> fold -> eval -> loop.

Anchored to the round-0 original throughout (KL reference = adapters/gates off).
`--fast-dev-run` runs the whole thing on the tiny-random model. See spec.md.
"""

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


def _mean_finite(xs) -> float:
    xs = [x for x in xs if x == x and x != float("inf")]
    return sum(xs) / len(xs) if xs else float("nan")


def steer_heal(model, tok, cfg: RunConfig, run_dir: Path) -> dict:
    hist_specs = []      # AdapterSpec per folded round (gated bake history)
    v0_flat = None       # round-0 direction, for the Q3 cosine
    rounds = []
    for rnd in range(cfg.n_rounds):
        logger.info(f"\n\n=== ROUND {rnd} [{cfg.model.split('/')[-1]} reg={cfg.reg}] gpu {gpu_mem()} ===")
        # extract teacher vector + sweep-generate steered data from the CURRENT student
        with baked(model, hist_specs):
            v = teacher_vec(model, tok, cfg)
            comps = generate_steered(model, tok, v, cfg)
        # filter under the ORIGINAL (no history, no steering) -- this picks the usable C
        logger.info(f"\n=== FILTER [{len(comps)} completions] gpu {gpu_mem()} ===")
        kept, scored = filter_completions(model, tok, comps, cfg)
        log_event(run_dir, stage="gen", round=rnd, n_comps=len(comps), n_kept=len(kept), scored=scored)

        # heal one round on top of the baked history, then fold
        logger.info(f"\n=== HEAL [{cfg.reg}] gpu {gpu_mem()} ===")
        lora, spec = heal_round(model, tok, kept, hist_specs, cfg)
        lora.save(str(run_dir / "ckpt" / f"r{rnd}.safetensors"), extra_meta={"round": str(rnd), "reg": cfg.reg})
        hist_specs.append(spec)

        # eval the student (all rounds baked) + Q1: trained-adapter output coherence
        logger.info(f"\n=== EVAL [tinymfv classic] gpu {gpu_mem()} ===")
        with baked(model, hist_specs):
            m = evaluate_model(model, tok, cfg)
            adapter = generate_plain(model, tok, cfg, n=min(6, cfg.n_prompts))
        adapter_ppl = _mean_finite([ppl_under_base(model, tok, a["prompt"], a["completion"]) for a in adapter])
        steered_ppl = _mean_finite([s["ppl"] for s in scored])
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
               "adapter_ppl": adapter_ppl, "n_kept": len(kept)}
        rounds.append(rec)
        log_event(run_dir, stage="round", **rec)
        logger.info(f"round {rnd}: auth_nats↓={m['auth_nats']:+.2f} care_nats={m['care_nats']:+.2f} "
                    f"coh→={m['coherence']:.3f} cos_v0={cos_v0:+.2f} adapter_ppl={adapter_ppl:.0f}")

    _log_loop_summary(rounds)
    write_map(run_dir, rounds)
    return rounds[-1]


def _log_loop_summary(rounds: list[dict]) -> None:
    from tabulate import tabulate
    # (rec_key, display header with direction arrow) -- single source of truth.
    cols = [("round", "round"), ("auth_nats", "auth_nats↓"), ("care_nats", "care_nats"),
            ("coherence", "coherence→"), ("cos_v0", "cos_v0→"),
            ("adapter_ppl", "adapter_ppl↓"), ("n_kept", "n_kept")]
    logger.info(
        "\nloop columns:\n"
        "   auth_nats↓ = Authority logp on Authority vignettes, NATS (TARGET: down = less deference)\n"
        "    care_nats = Care logp, NATS (off-target axis -- should move LESS than auth if surgical)\n"
        "   coherence→ = p_any_ans = mean_pmass_allowed (OFF-TARGET: hold ~1.0)\n"
        "      cos_v0→ = cosine of round vector vs round-0 vector (direction stability)\n"
        " adapter_ppl↓ = ppl-under-original of the no-steering adapter generations"
    )
    logger.info(
        "\nSHOULD (Q2 loop-coherent): coherence stays >= round-0 floor across rounds (heal holds it up). "
        "If coherence falls each round, the loop accumulates incoherency faster than heal removes it.\n"
        "SHOULD (Q3 direction): auth_nats FALLS monotonically (0.5-2 nats is a real shift) and cos_v0 "
        "stays > 0.5. If care_nats falls as much as auth_nats, it's broad permissivizing not surgical."
    )
    tbl = [{disp: r.get(key) for key, disp in cols} for r in rounds]
    logger.info("\nloop summary:\n" + tabulate(tbl, headers="keys", tablefmt="github", floatfmt=".3f") + "\n")


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
