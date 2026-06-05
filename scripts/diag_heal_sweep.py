"""Fast healing-hypothesis sweep, the RIGHT way: from the round-(N-1) CHECKPOINT.

The earlier diag_barrier.py re-healed a FRESH adapter from BASE (hist=[]), so the kl barrier
anchored to base and never saw the loop state. This loads the real round-0 checkpoint as baked
history, re-heals round-1's kept data on top, and varies ONLY the regulariser + the barrier
REFERENCE. That isolates: at round 1 (where the loop starts degenerating), which regulariser adds
the most NEW trait at the least coherence cost?

The decisive contrast is kl_rev ref=base vs ref=prev:
  ref=base  -> KL(student || ORIGINAL). The student already carries round-0's trait, so this leashes
               it back toward base and partly UNDOES the prev round.
  ref=prev  -> KL(student || prev-round student). Penalises only THIS round's new divergence = a
               trust region, so trait accumulates while each step stays coherent.

Metric: dAuth vs PREV (= new trait this round, the thing we want negative) at coherence >= prev.

Run: uv run python scripts/diag_heal_sweep.py out/20260604T231906_gemma-3-4b-it_nll_s42 1
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
from steer_heal.run import setup_logging  # noqa: E402
from steer_heal.ws.bake import AdapterSpec, baked  # noqa: E402

setup_logging()  # INFO -> stdout via tqdm.write, DEBUG (per-step bake trace) -> logs/*_verbose.log
run_dir = Path(sys.argv[1])
gen_round = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # re-heal THIS round's data on r0..r(N-1) history
base_cfg = RunConfig()

# REGULARISER ABLATION (reg, lam, tau, ref, wd), all at ref=prev (the base-vs-prev DIRECTION
# question is settled: ref=base undoes prev, confirmed in #97 -- nll +1.157, kl_rev/base +0.855).
# Hold the reference fixed and ask which REGULARISER best trades coherence for trait, by the
# cohΔ/authΔ headline. Five families: nll (no reg, control), wd (Frobenius shrink via AdamW),
# kl_rev (mode-seeking trust region), kl_fwd (mass-covering), spectral_norm (operator-norm penalty).
#
# This is the FULL authoritative grid. The `# [#98]`-tagged rows are commented out because pueue 98
# already produced them -- uncomment to re-run from scratch. The active rows are the widened ends
# #98 never ran (the gap-fill, pueue 99). The combined table is read from BOTH logs. wd<=15 was
# byte-identical to the no-reg control (inert) so it stays commented; wd=30 moved trait MORE
# (dAuth_base -0.997 vs -0.782) AND held coherence, so wd 60/120 trace the curve above the knee.
GRID = [
    # ("nll",           0.0,  0.0, "prev",   0.0),  # [#98] control: pure SFT, no reg
    # ("nll",           0.0,  0.0, "prev",  15.0),  # [#98] INERT: byte-identical to wd=0 (decay too small to bite)
    # ("nll",           0.0,  0.0, "prev",  30.0),  # [#98] wd at the knee (AdamW Frobenius shrink on ΔW)
    ("nll",           0.0,  0.0, "prev",  60.0),  # wd above knee -- does coherence keep improving?
    ("nll",           0.0,  0.0, "prev", 120.0),  # wd strong -- where does trait start to erode?
    ("kl_rev",        0.03, 0.5, "prev",   0.0),  # mode-seeking trust region, gentle (#82 best-retain end)
    ("kl_rev",        0.05, 0.5, "prev",   0.0),  # between 0.03 and 0.1: does the slope peak below 0.1? (#98: 0.1 beat 0.3)
    # ("kl_rev",        0.1,  0.5, "prev",   0.0),  # [#98] mode-seeking trust region, mid (current front-runner -0.13)
    # ("kl_rev",        0.3,  0.5, "prev",   0.0),  # [#98] stronger trust region
    ("kl_rev",        1.0,  0.5, "prev",   0.0),  # strong (#82: over-tight, undoes trait) -- the bracket end
    # ("kl_fwd",        0.1,  0.5, "prev",   0.0),  # [#98] mass-covering, gentle
    ("kl_fwd",        0.3,  0.5, "prev",   0.0),  # mass-covering, stronger (expect: dilutes trait)
    # spectral_norm is no longer a reg -- it's the independent cfg.spectral_lam knob now (composes with
    # kl_rev). #98 swept it as reg=spectral_norm (0.01/0.1/1.0); to redo, set spectral_lam, not reg.
]
logger.info(f"heal sweep from round-{gen_round-1} checkpoint, re-heal round-{gen_round} data: {len(GRID)} configs")

tok = AutoTokenizer.from_pretrained(base_cfg.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_cfg.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
).eval()

# baked history = the real round-0..round-(gen_round-1) adapters from the source run.
hist_specs = [AdapterSpec.from_checkpoint(model, str(run_dir / "ckpt" / f"r{i}.safetensors"))
              for i in range(gen_round)]
logger.info(f"loaded {len(hist_specs)} history checkpoint(s): r0..r{gen_round-1}")

# round-gen_round kept completions = the data round gen_round actually trained on.
gen = next(e for e in srsly.read_jsonl(run_dir / "events.jsonl")
           if e["stage"] == "gen" and e["round"] == gen_round)
kept = [{"prompt": s["prompt"], "completion": s["completion"]} for s in gen["scored"] if s["keep"]]
logger.info(f"loaded {len(kept)} kept completions from round {gen_round}")

base_m = evaluate_model(model, tok, base_cfg)
with baked(model, hist_specs):
    prev_m = evaluate_model(model, tok, base_cfg)  # round-(gen_round-1) HEALED = the start point this round must improve on
logger.info(f"base: auth={base_m['auth_nats']:+.4f} coh={base_m['coherence']:.5f}")
logger.info(f"prev (r{gen_round-1} healed): auth={prev_m['auth_nats']:+.4f} coh={prev_m['coherence']:.5f}")
logger.info("SHOULD: dAuth_vs_prev NEGATIVE = this round ADDED trait; POSITIVE = the barrier UNDID prev. "
            "ref=base should undo (>=0) where ref=prev adds (<0), at coherence >= prev.")

rows = []
for reg, lam, tau, ref, wd in GRID:
    cfg = dataclasses.replace(base_cfg, reg=reg, lam=lam, tau=tau, barrier_ref=ref, weight_decay=wd)
    torch.manual_seed(cfg.seed)  # identical LoRA-A init across configs -> only the regulariser differs
    lora, spec, heal_nll = heal_round(model, tok, kept, hist_specs, cfg)
    with baked(model, hist_specs + [spec]):  # full round-gen_round student = history + this round's adapter
        m = evaluate_model(model, tok, cfg)
    dAuth_base = m["auth_nats"] - base_m["auth_nats"]
    dCoh_base = m["coherence"] - base_m["coherence"]
    dAuth_prev = m["auth_nats"] - prev_m["auth_nats"]
    dCoh_prev = m["coherence"] - prev_m["coherence"]
    # THE HEADLINE: coherence cost per unit of trait, the trade-off slope dCoh/dAuth.
    # We want trait to move (dAuth NEGATIVE) at little coherence cost (dCoh ~0), so a GOOD
    # config has a small-magnitude ratio (or negative = free coherence). NaN-guard the
    # denominator: a config that barely moves auth (|dAuth|<0.05 noise floor) makes the
    # ratio explode/flip sign on noise, so it is not a meaningful efficiency -- blank it.
    eps = 0.05
    # HEADLINE scaled x100 so the tiny coherence-per-trait slope keeps resolving digits under
    # the table's +.4f (raw ~+0.001 -> "+0.0011", and +0.0001 -> "+0.0001" both collapse to the
    # noise floor; x100 -> "+0.1100" vs "+0.0100" stays distinguishable). Units: centinats coh / nat auth.
    coh_per_auth_base = 100 * dCoh_base / dAuth_base if abs(dAuth_base) > eps else float("nan")
    coh_per_auth_prev = 100 * dCoh_prev / dAuth_prev if abs(dAuth_prev) > eps else float("nan")
    rows.append({  # HEADLINE first. Direction matters (NOT abs): most-NEGATIVE best = trait moved AND
        "cohΔ/authΔ_base×100↓": coh_per_auth_base,  # coh ROSE (free lunch); then small positive = cheap.
        "cohΔ/authΔ_prev×100↓": coh_per_auth_prev,
        "reg": reg, "lam": lam, "tau": tau, "ref": ref, "wd": wd,
        "auth↓": m["auth_nats"], "dAuth_base↓": dAuth_base, "dAuth_prev↓": dAuth_prev,
        "coh↑": m["coherence"], "dCoh_base↑": dCoh_base, "dCoh_prev↑": dCoh_prev, "heal_nll↓": heal_nll,
    })
    logger.info(f"  {reg} lam={lam} tau={tau} ref={ref} wd={wd}: "
                f"cohΔ/authΔ_base×100={coh_per_auth_base:+.4f}  auth={m['auth_nats']:+.4f} "
                f"dAuth_base={dAuth_base:+.4f} dAuth_prev={dAuth_prev:+.4f} coh={m['coherence']:.5f}")

# bookend reference rows so the swept configs read against base (origin) and prev (r0 healed = the
# anchor this round starts from). Every config sits BETWEEN these two; prev's own slope shows what r0's
# heal achieved (the bar to reproduce a round deeper).
for tag, mm in (("(prev=r0heal)", prev_m), ("(base origin)", base_m)):
    dAb, dCb = mm["auth_nats"] - base_m["auth_nats"], mm["coherence"] - base_m["coherence"]
    dAp, dCp = mm["auth_nats"] - prev_m["auth_nats"], mm["coherence"] - prev_m["coherence"]
    rows.append({
        "cohΔ/authΔ_base×100↓": 100 * dCb / dAb if abs(dAb) > 0.05 else float("nan"),
        "cohΔ/authΔ_prev×100↓": 100 * dCp / dAp if abs(dAp) > 0.05 else float("nan"),
        "reg": tag, "lam": "", "tau": "", "ref": "", "wd": "",
        "auth↓": mm["auth_nats"], "dAuth_base↓": dAb, "dAuth_prev↓": dAp,
        "coh↑": mm["coherence"], "dCoh_base↑": dCb, "dCoh_prev↑": dCp, "heal_nll↓": float("nan"),
    })

print(f"\nheal sweep from r{gen_round-1} checkpoint, re-heal r{gen_round} data (vary regulariser + barrier ref only):")
print("HEADLINE = cohΔ/authΔ×100: centinats of coherence lost per nat of trait moved (the heal slope).")
print("  DIRECTION, not magnitude: most-NEGATIVE is best (trait moved AND coherence ROSE = free lunch);")
print("  then small-positive = cheap; large-positive = trait cost a lot of coherence. Sorted best-first.")
print("  blank ratio = |dAuth|<0.05 (config barely moved trait; slope is noise, not an efficiency).")
print("dAuth_prev = NEW trait this round (NEGATIVE = added); ref=base vs prev is the direction crux.\n")
# sort by the signed slope (NOT abs): most-negative free-lunch row first, NaN (do-nothing) last.
rows.sort(key=lambda r: (r["cohΔ/authΔ_base×100↓"] if r["cohΔ/authΔ_base×100↓"] == r["cohΔ/authΔ_base×100↓"] else 1e9))
# per-column precision: headline x100 + coherence deltas get the extra digits that discriminate close
# configs; reg/lam/tau/wd stay compact. Tuple order matches the rows-dict key order above.
fmt = ("+.4f", "+.4f", "g", "g", "g", "g", "g", "+.4f", "+.4f", "+.4f", ".5f", "+.5f", "+.5f", "+.3f")
print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=fmt))
print(f"\nbase auth={base_m['auth_nats']:+.3f} coh={base_m['coherence']:.3f} | "
      f"prev(r{gen_round-1}) auth={prev_m['auth_nats']:+.3f} coh={prev_m['coherence']:.3f} | source {run_dir.name}")
