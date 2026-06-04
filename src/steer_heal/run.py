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
from steer_heal.filter import filter_completions
from steer_heal.heal import heal_round
from steer_heal.io import append_result, log_event, make_run_dir
from steer_heal.plot import write_map
from steer_heal.steering import generate_steered, teacher_vec
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
    logger.info(f"loaded {model_id} (dtype={dtype}, attn={attn}, layers={model.config.num_hidden_layers})")
    return model, tok


def _flatten_v(v) -> torch.Tensor:
    return torch.cat([v.state[li]["v"].flatten().float() for li in sorted(v.state)])


def steer_heal(model, tok, cfg: RunConfig, run_dir: Path) -> dict:
    hist_specs = []      # AdapterSpec per folded round (gated bake history)
    v0_flat = None       # round-0 direction, for the Q3 cosine
    rounds = []
    for rnd in range(cfg.n_rounds):
        logger.info(f"\n=== ROUND {rnd} [{cfg.model.split('/')[-1]} reg={cfg.reg}] ===")
        # extract teacher vector + generate steered data from the CURRENT student
        with baked(model, hist_specs):
            v = teacher_vec(model, tok, cfg)
            comps = generate_steered(model, tok, v, alpha=1.0, cfg=cfg)
        # filter under the ORIGINAL (no history, no steering)
        kept, scored = filter_completions(model, tok, comps, cfg)
        log_event(run_dir, stage="gen", round=rnd, n_comps=len(comps), n_kept=len(kept), scored=scored)

        # heal one round on top of the baked history, then fold
        lora, spec = heal_round(model, tok, kept, hist_specs, cfg)
        lora.save(str(run_dir / "ckpt" / f"r{rnd}.safetensors"), extra_meta={"round": str(rnd), "reg": cfg.reg})
        hist_specs.append(spec)

        # eval the student (all rounds baked)
        with baked(model, hist_specs):
            m = evaluate_model(model, tok, cfg)

        vf = _flatten_v(v)
        v0_flat = vf if v0_flat is None else v0_flat
        cos_v0 = float(cosine_similarity(vf, v0_flat, dim=0))
        rec = {"round": rnd, **m, "cos_v0": cos_v0, "c_star": float(v.cfg.coeff), "n_kept": len(kept)}
        rounds.append(rec)
        log_event(run_dir, stage="round", **rec)
        logger.info(f"round {rnd}: auth={m['auth']:.3f} care={m['care']:.3f} "
                    f"coh={m['coherence']:.3f} cos_v0={cos_v0:+.2f}")

    map_path = write_map(run_dir, rounds)
    logger.info(f"map: {map_path}")
    return rounds[-1]


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
