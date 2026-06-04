"""Q1 trait-persistence: does the trained adapter move the profile AWAY from base,
or is it a coherent no-op (healed == reverted to base)?

adapter_ppl < steered_ppl only says the adapter is coherent. A do-nothing adapter
is also coherent. The distinguishing check is the profile DELTA vs base: if
heal kept the trait, socialnorms/care shift in the steering direction while
coherence holds. If base == adapter, the adapter learned nothing.

Run: uv run python scripts/diag_heal.py out/<ts>_<slug>/ckpt/r0.safetensors
"""

import sys

import torch
import tinymfv
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from steer_heal.config import RunConfig  # noqa: E402
from steer_heal.steering import teacher_vec  # noqa: E402
from steer_heal.ws.bake import AdapterSpec, baked  # noqa: E402

ckpt = sys.argv[1]
N_VIG = None if (len(sys.argv) > 2 and sys.argv[2] == "all") else int(sys.argv[2]) if len(sys.argv) > 2 else 24
NO_STEER = "--no-steer" in sys.argv
cfg = RunConfig(n_prompts=12)
MODEL = cfg.model

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
).eval()


def profile(label):
    rep = tinymfv.evaluate(model, tok, name="classic", n_vignettes=N_VIG,
                           conditions=("other_violate",), max_think_tokens=cfg.eval_think_tokens, device=model.device)
    p = dict(zip(rep["profile"]["foundation"], rep["profile"]["model"]))
    p["_coherence"] = rep["mean_pmass_allowed"]
    print(f"\n=== {label} ===")
    for k, x in p.items():
        print(f"  {k:12s} {x:.4f}")
    return p


# three points: base, in-band steered (the raw teacher), trained adapter.
print(f"n_vignettes={N_VIG} no_steer={NO_STEER} ckpt={ckpt}")
base = profile("BASE (no adapter)")

if NO_STEER:
    steer = base  # skip the slow steered eval; deltas vs steer are then 0
else:
    v = teacher_vec(model, tok, cfg)
    with v(model, C=v.cfg.coeff):
        steer = profile(f"STEERED (raw, c={v.cfg.coeff:.1f})")

spec = AdapterSpec.from_checkpoint(model, ckpt)
with baked(model, [spec]):
    adapt = profile(f"ADAPTER (r0, baked) {ckpt.split('/')[-1]}")

# SHOULD: adapter delta has the SAME SIGN as steered delta on the trait axis
# (socialnorms, care) -> heal kept the trait. If adapter delta ~ 0 -> no-op
# (we "healed" by reverting to base). Coherence: steered may drop, adapter holds.
print("\n=== trait axis: did the adapter keep the steering direction? ===")
print(f"  {'foundation':12s} {'base':>8s} {'steer':>8s} {'adapt':>8s}  "
      f"{'d_steer':>8s} {'d_adapt':>8s}  same_sign")
keys = [k for k in base if not k.startswith("_")]
for k in sorted(keys, key=lambda k: -abs(steer[k] - base[k])):
    ds, da = steer[k] - base[k], adapt[k] - base[k]
    same = "YES" if (ds * da > 0 and abs(da) > 0.01) else ("no-op" if abs(da) < 0.01 else "OPPOSITE")
    print(f"  {k:12s} {base[k]:+8.3f} {steer[k]:+8.3f} {adapt[k]:+8.3f}  "
          f"{ds:+8.3f} {da:+8.3f}  {same}")
print(f"  {'coherence':12s} {base['_coherence']:8.3f} {steer['_coherence']:8.3f} "
      f"{adapt['_coherence']:8.3f}")
