"""Diagnostic: does the steering vector move the moral-foundation profile, and where?

Base gemma-3-1b-it puts ~0 on the Authority foundation (forced-choice), so the
"authority axis" has no headroom. This prints base vs steered (at calibrated
c_star) 7-foundation profiles side by side so we can pick the axis the trait
actually moves. Run: uv run python scripts/diag_axis.py
"""

import sys

import torch
import tinymfv
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from steer_heal.config import RunConfig  # noqa: E402
from steer_heal.steering import teacher_vec  # noqa: E402

MODEL = "google/gemma-3-1b-it"
cfg = RunConfig(model=MODEL, n_prompts=12)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
).eval()

v = teacher_vec(model, tok, cfg)


def profile(label):
    rep = tinymfv.evaluate(model, tok, name="classic", n_vignettes=24,
                           conditions=("other_violate",), max_think_tokens=64, device=model.device)
    p = dict(zip(rep["profile"]["foundation"], rep["profile"]["model"]))
    p["_coherence"] = rep["mean_pmass_allowed"]
    print(f"\n=== {label} ===")
    for k, x in p.items():
        print(f"  {k:12s} {x:.4f}")
    return p


base = profile("BASE (c=0)")
with v(model, C=v.cfg.coeff):
    steer = profile(f"STEERED (c_star={v.cfg.coeff:.1f}, ~1 nat)")

print("\n=== delta (steered - base), sorted by |Δ| ===")
keys = [k for k in base if not k.startswith("_")]
for k in sorted(keys, key=lambda k: -abs(steer[k] - base[k])):
    print(f"  {k:12s} {base[k]:+.4f} -> {steer[k]:+.4f}   Δ={steer[k]-base[k]:+.4f}")
print(f"  coherence    {base['_coherence']:.3f} -> {steer['_coherence']:.3f}")
