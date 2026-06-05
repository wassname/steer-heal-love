"""Barrier-strength sweep: is the trait failing to transfer because the barrier (kl/wd) is too
strong, or because the kept data is near-base? Re-heal from ONE run's cached kept completions with
the SAME init seed, varying ONLY (reg, lam, tau). Same data + same init => the only thing that moves
healed auth_nats is the barrier.

reg=nll is the ablation: barrier OFF. If nll ALSO lands near base, the data is the ceiling, not the
barrier. If nll (or weak kl/wd) retains MORE trait than kl_rev lam=1.0, the barrier was killing it.

Run: uv run python scripts/diag_barrier.py out/20260604T194133_gemma-3-4b-it_kl_rev_s42/
"""
import dataclasses
import sys
from pathlib import Path

import srsly
import torch
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from steer_heal.config import RunConfig  # noqa: E402
from steer_heal.eval import evaluate_model  # noqa: E402
from steer_heal.heal import heal_round  # noqa: E402
from steer_heal.ws.bake import baked  # noqa: E402

run_dir = Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "barrier"  # "barrier" (kl sweep) or "wd" (decay decade sweep)
gen_round = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # which round's kept data to re-heal (0 = clean; later = messier)
base_cfg = RunConfig()

# (reg, lam, tau) grids. nll = barrier off (ablation) and the shared trait-ceiling reference.
GRIDS = {
    # kl_rev strength + a tau probe. lam 0.03 (w2s) .. 1.0 (current default).
    "barrier": [
        ("nll",    0.0,  0.5),   # ablation: no barrier at all
        ("kl_rev", 0.03, 0.5),
        ("kl_rev", 0.1,  0.5),
        ("kl_rev", 0.3,  0.5),
        ("kl_rev", 1.0,  0.5),
        ("kl_rev", 0.3,  1.0),   # weaker via higher tau (engages later)
    ],
    # pure linear kl_rev: tau=0 => barrier = lam*relu(div) = lam*div, always on, no deadband
    # (the w2s form). Cleaner knob than the hinge; compare against the tau=0.5 rows in "barrier".
    "tau0": [
        ("nll",    0.0,  0.0),
        ("kl_rev", 0.03, 0.0),
        ("kl_rev", 0.1,  0.0),
        ("kl_rev", 0.3,  0.0),
        ("kl_rev", 1.0,  0.0),
    ],
    # tau sweep: fix lam (middling barrier) and vary the deadband tau. Higher tau = barrier engages
    # only on larger divergence = weaker. Shows whether a deadband helps on degenerate (round 2) data.
    "tau": [
        ("nll",    0.0, 0.0),
        ("kl_rev", 0.3, 0.0),
        ("kl_rev", 0.3, 0.25),
        ("kl_rev", 0.3, 0.5),
        ("kl_rev", 0.3, 1.0),
        ("kl_rev", 0.3, 2.0),
    ],
    # weight decay: a WEIGHTS-space constraint (AdamW decoupled decay, tau irrelevant). Its per-step
    # shrink is lr*wd, and lr~1e-4 is tiny, so #82 found wd<=0.1 byte-identical to nll (~0.1% shrink
    # over 252 steps). Sweep up to 100 to find where cumulative shrink (252*lr*wd) reaches order-1.
    "wd": [
        ("nll", 0.0,  0.5),
        ("wd",  1e-1, 0.5),
        ("wd",  1.0,  0.5),
        ("wd",  3.0,  0.5),
        ("wd",  10.0, 0.5),
        ("wd",  30.0, 0.5),
        ("wd",  100.0, 0.5),
    ],
}
GRID = GRIDS[mode]
logger.info(f"barrier sweep mode={mode}: {len(GRID)} configs")

# kept completions (keep==True) from a CHOSEN round of the source run. round 0 = clean steered-on-base
# data; later rounds = data after the loop started degenerating (repetition), the regime where the
# barrier is hypothesised to matter (it was pure-cost on clean round-0 data, #82/85/86).
gen = next(e for e in srsly.read_jsonl(run_dir / "events.jsonl")
           if e["stage"] == "gen" and e["round"] == gen_round)
kept = [{"prompt": s["prompt"], "completion": s["completion"]} for s in gen["scored"] if s["keep"]]
logger.info(f"loaded {len(kept)} kept completions from {run_dir.name} round {gen_round}")

tok = AutoTokenizer.from_pretrained(base_cfg.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_cfg.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
).eval()

base_m = evaluate_model(model, tok, base_cfg)
logger.info(f"base: auth_nats={base_m['auth_nats']:+.3f} care_nats={base_m['care_nats']:+.3f} coh={base_m['coherence']:.3f}")

rows = []
for reg, lam, tau in GRID:
    # "wd" grid rows are now a weights-space knob, not a reg value: map to reg=nll + weight_decay=lam.
    if reg == "wd":
        cfg = dataclasses.replace(base_cfg, reg="nll", lam=0.0, tau=0.0, weight_decay=lam)
    else:
        cfg = dataclasses.replace(base_cfg, reg=reg, lam=lam, tau=tau, weight_decay=0.0)
    torch.manual_seed(cfg.seed)  # identical LoRA-A init across barrier values -> only the barrier differs
    lora, spec, heal_nll = heal_round(model, tok, kept, [], cfg)
    with baked(model, [spec]):
        m = evaluate_model(model, tok, cfg)
    dauth = m["auth_nats"] - base_m["auth_nats"]
    dcoh = m["coherence"] - base_m["coherence"]
    # ONE strength knob per row: kl-barrier weight for kl_rev/kl_fwd, AdamW weight_decay for wd,
    # ignored for nll. tau (kl deadband) only applies to the kl regs -> "-" otherwise.
    is_kl = reg in ("kl_rev", "kl_fwd")
    rows.append({"reg": reg, "strength": lam, "tau(kl only)": (f"{tau:.1f}" if is_kl else "-"),
                 "heal_nll↓": heal_nll, "auth_nats↓": m["auth_nats"], "dAuth↓": dauth,
                 "care_nats": m["care_nats"], "coh→": m["coherence"], "dCoh": dcoh})
    logger.info(f"  {reg} strength={lam}{f' tau={tau}' if is_kl else ''}: "
                f"auth={m['auth_nats']:+.3f} (dAuth={dauth:+.3f}) coh={m['coherence']:.3f}")

logger.info("SHOULD: if nll/weak-barrier retain MORE trait (more negative dAuth) at similar coh, the "
            "barrier was killing the trait. If ALL rows sit near dAuth~0, the kept data is near-base.")
print("\nbarrier sweep (re-heal #79 kept data, vary the regulariser only; dAuth/dCoh vs base):")
print("strength = kl-barrier weight (kl_rev) OR AdamW weight_decay (wd); tau = kl deadband, n/a for wd/nll\n")
print(tabulate(rows, headers="keys", tablefmt="github", floatfmt="+.3f"))
print(f"\nbase auth_nats={base_m['auth_nats']:+.3f} coh={base_m['coherence']:.3f} | source {run_dir.name}")
