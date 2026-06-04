"""steer_heal pipeline entry point.

Loop (anchored to the round-0 original throughout, see spec.md):
  teacher vector (steering-lite) -> iso-KL dose -> generate -> U1 filter
  -> heal one round (SFT + KL-rev-to-original barrier) -> fold (gated bake)
  -> tinymfv eval -> repeat.

Stages marked TODO are ported from docs/vendor/* as we implement them; this
file fails fast at the first unimplemented stage rather than stubbing fake
behaviour. `--fast-dev-run` runs the whole thing on the tiny-random model.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import tyro
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from steer_heal.config import RunConfig, resolve

REPO = Path(__file__).resolve().parents[2]


def setup_logging() -> None:
    # tqdm-safe loguru, single-char level icons, verbose copy on disk.
    logger.remove()
    logger.add(lambda m: tqdm.write(m, end=""), colorize=True,
               format="<level>{level.icon}</level> {message}", level="INFO")
    for lvl, ic in [("INFO", "I"), ("WARNING", "W"), ("ERROR", "E"), ("DEBUG", "D")]:
        logger.level(lvl, icon=ic)
    log_dir = REPO / "logs"
    log_dir.mkdir(exist_ok=True)
    f = log_dir / f"{datetime.now():%Y%m%dT%H%M%S}_verbose.log"
    logger.add(f, format="{time:HH:mm:ss} | {level: <7} | {name}:{function}:{line} - {message}",
               level="DEBUG")
    logger.info(f"verbose log: {f}")


def load_model(model_id: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    attn = os.environ.get("STEER_ATTN_IMPL", "eager")  # set =flash_attention_2 on real runs
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation=attn,
    )
    model.eval()
    logger.info(f"loaded {model_id} (dtype={dtype}, attn={attn})")
    return model, tok


# ── stages (ported from docs/vendor as we implement; fail fast until then) ──

def teacher_vec(model, tok, cfg: RunConfig):
    # steering-lite Vector.train(pos=trait-sysprompt, neg=neutral-sysprompt) @ assistant tag,
    # then .calibrate(target_kl=cfg.target_kl). See docs/vendor/steering-lite + isokl.
    raise NotImplementedError("TODO: teacher_vec via steering-lite + iso-KL calibration")


def generate_and_filter(model, tok, v, orig, cfg: RunConfig):
    # gen at alpha*c_star (steering-lite hook); keep coherent & enact-not-narrate (U1).
    raise NotImplementedError("TODO: generate_and_filter (U1 gate)")


def heal(model, orig, comps, cfg: RunConfig):
    # SFT + lam*relu(div - tau); div in {nll, kl_fwd, kl_rev, wd}; KL ref = orig (gates off).
    # adapter + gated bake ported from docs/vendor/w2schar-mini/src/csm/ws.
    raise NotImplementedError("TODO: heal (U2 barrier) + fold via w2schar ws.bake")


def evaluate(model, cfg: RunConfig) -> dict:
    # tinymfv auth/care axes + p_ans_any/json_is_valid/ppx_json.
    raise NotImplementedError("TODO: tinymfv eval + plotly map (port csm/plot.py _build_scatter)")


def steer_heal(model, tok, orig, cfg: RunConfig):
    for r in range(cfg.n_rounds):
        logger.info(f"── round {r} ──")
        v = teacher_vec(model, tok, cfg)
        comps = generate_and_filter(model, tok, v, orig, cfg)
        heal(model, orig, comps, cfg)
        logger.info(evaluate(model, cfg))
    return model


def main(cfg: RunConfig) -> None:
    setup_logging()
    cfg = resolve(cfg)
    torch.manual_seed(cfg.seed)
    logger.info(f"config: {cfg}")
    dtype = getattr(torch, cfg.dtype)
    model, tok = load_model(cfg.model, dtype)
    orig = model  # round-0 anchor; KL reference = same module with adapter gates off
    steer_heal(model, tok, orig, cfg)


if __name__ == "__main__":
    main(tyro.cli(RunConfig))
