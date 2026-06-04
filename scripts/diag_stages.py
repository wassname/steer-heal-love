"""Target vs off-target effect at each stage, all at the SAME n_vignettes.

TARGET    = Authority foundation, want DOWN (trait = "do not defer to authority").
            (also report SocialNorms + Care, the axis the 1b note flagged.)
OFF-TARGET= coherence = tinymfv mean_pmass_allowed = p_any_ans, want HELD ~1.0.

Stages: base -> steered(c=0.5,1.0) -> one row per adapter ckpt (labeled by its
reg). One model load, one vignette set, so every row is paired and comparable.

Run: uv run python scripts/diag_stages.py <ckpt1> [ckpt2 ...] [n|all]
"""

import json
import sys
from pathlib import Path

import torch
import tinymfv
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from steer_heal.config import RunConfig  # noqa: E402
from steer_heal.eval import foundation_nats  # noqa: E402
from steer_heal.steering import teacher_vec  # noqa: E402
from steer_heal.ws.bake import AdapterSpec, baked  # noqa: E402

# Trailing "all"/int is the vignette count; everything else is a ckpt path.
argv = sys.argv[1:]
N_VIG = None
if argv and (argv[-1] == "all" or argv[-1].isdigit()):
    N_VIG = None if argv[-1] == "all" else int(argv[-1])
    argv = argv[:-1]
ckpts = argv  # 1+ adapter checkpoints


def ckpt_label(path: str) -> str:
    """Row label = the run's reg (kl_rev/nll/...) from metadata.json two dirs up."""
    m = json.load(open(Path(path).parents[1] / "metadata.json"))
    reg = m.get("cfg", m).get("reg", "?")
    return f"heal_{reg}"


cfg = RunConfig(n_prompts=12)

tok = AutoTokenizer.from_pretrained(cfg.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    cfg.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
).eval()


def prof():
    rep = tinymfv.evaluate(model, tok, name="classic", n_vignettes=N_VIG,
                           conditions=("other_violate",), max_think_tokens=cfg.eval_think_tokens,
                           device=model.device, return_per_row=True)
    p = foundation_nats(rep)  # logp per foundation, NATS
    p["coherence"] = rep["mean_pmass_allowed"]
    return p


v = teacher_vec(model, tok, cfg)
adapters = [(ckpt_label(p), AdapterSpec.from_checkpoint(model, p)) for p in ckpts]

rows = {}
rows["base"] = prof()
for c in (0.5, 1.0):  # 0.5 = coherent operating point; 1.0 = the collapse end
    with v(model, C=c * v.cfg.coeff):
        rows[f"steered(c={c:g})"] = prof()
for label, spec in adapters:
    with baked(model, [spec]):
        rows[label] = prof()

# target = Authority log p (down good, NATS), off-target = coherence (held good).
# THE Gate-3 question (user): is the trained adapter more coherent PER UNIT behaviour
# change than raw steering? -> coh_cost = |dCoh| / |dAuth| (coherence lost per nat of
# Authority shift). LOWER = better pareto. If an adapter has lower coh_cost than the
# steered rows, distill+heal bought a better behaviour/coherence trade than steering.
b = rows["base"]
d_auth_steer = rows["steered(c=0.5)"]["Authority"] - b["Authority"]  # retain denom = operating-point shift
print(f"\nn_vignettes={N_VIG}  TARGET=Authority log p (NATS, want DOWN)  OFF-TARGET=coherence (want ~{b['coherence']:.2f})")
print("All foundation columns in NATS (log p, choice-logprob). retain = dAuth(stage)/dAuth(steered c=0.5): "
      "1=heal kept the operating-point trait, 0=reverted to base (UNDO), <0=wrong way.")
print("coh_cost = |dCoh|/|dAuth| = coherence lost per nat of behaviour change. LOWER is a BETTER pareto. "
      "The point of distill+heal: adapter coh_cost < steered coh_cost. SHOULD: a real HEAL keeps |dAuth| "
      "(retain>0) at near-zero |dCoh| (low coh_cost); an UNDO has retain~0 (no trait, nothing to cost).")
tbl = []
for stage, p in rows.items():
    dA = p["Authority"] - b["Authority"]
    dC = p["coherence"] - b["coherence"]
    retain = dA / d_auth_steer if abs(d_auth_steer) > 1e-6 else float("nan")
    coh_cost = abs(dC) / abs(dA) if abs(dA) > 1e-6 else float("nan")
    tbl.append({
        "stage": stage,
        "auth_nats↓": p["Authority"], "dAuth": dA, "retain": retain,
        "socnorm": p["SocialNorms"], "care": p["Care"],
        "coherence→": p["coherence"], "dCoh": dC, "coh_cost↓": coh_cost,
    })
print(tabulate(tbl, headers="keys", tablefmt="github", floatfmt="+.3f"))
