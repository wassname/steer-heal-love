"""GATE 1: does the steering vector move the target WHILE staying coherent?

Sweep steering strength c and, at each c, eval the foundation profile AND
generate one completion, so we judge both the metric and the text. This is the
gate that must pass before filter/lora gates matter: if no c gives a target shift
at coherence ~0.95, the vector is the problem, not the heal.

Reading the (dAuth, coherence) pareto:
  PASS         a c with large -dAuth at coherence >= ~0.95 (knee before collapse)
  too weak     -dAuth ~ 0 until coherence cliffs
  too strong   -dAuth only appears once coherence < ~0.85 (no knee, bad pareto)
  wrong target |dCare| or |dSocialNorms| > |dAuth| at the same c
  collapse     all foundations shrink proportionally + coherence drops (no specificity)

Run: uv run python scripts/diag_csweep.py [n|all]
"""

import sys

import torch
import tinymfv
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from steer_heal.config import RunConfig  # noqa: E402
from steer_heal.eval import foundation_nats  # noqa: E402
from steer_heal.prompts import POOL, chat_prompt  # noqa: E402
from steer_heal.steering import _gen_one, teacher_vec  # noqa: E402

N_VIG = None if (len(sys.argv) > 1 and sys.argv[1] == "all") else int(sys.argv[1]) if len(sys.argv) > 1 else None
CS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
cfg = RunConfig(n_prompts=12)

tok = AutoTokenizer.from_pretrained(cfg.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    cfg.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
).eval()

v = teacher_vec(model, tok, cfg)
demo_prompt = chat_prompt(tok, cfg.gen_system, POOL[0])  # fixed prompt for the qualitative read


def profile():
    rep = tinymfv.evaluate(model, tok, name="classic", n_vignettes=N_VIG,
                           conditions=("other_violate",), max_think_tokens=cfg.eval_think_tokens,
                           device=model.device, return_per_row=True)
    nats = foundation_nats(rep)  # logp per foundation, NATS
    nats["coherence"] = rep["mean_pmass_allowed"]
    return nats


rows, samples = [], []
for c in CS:
    with v(model, C=c * v.cfg.coeff):
        p = profile()
        gen = _gen_one(model, tok, demo_prompt, cfg)
    rows.append((c, p))
    samples.append((c, gen))

b = rows[0][1]
print(f"\nn_vignettes={N_VIG}  c-sweep of the teacher vector (coeff={v.cfg.coeff})  ALL VALUES IN NATS (log p, choice-logprob)")
print("auth_sep = base - steered Authority log p (POSITIVE = steered attributes authority-defiance "
      "less to authority = correct direction). Scale is tinymfv's diagonal log(p); base auth_nats "
      "~-2.3, a real shift is ~1-3 nats. NOT steering-lite's 0.5-2 p(is-wrong) metric.")
tbl = []
for c, p in rows:
    tbl.append({
        "c": c,
        "auth_nats↓": p["Authority"], "auth_sep↑": b["Authority"] - p["Authority"],
        "care_nats": p["Care"], "care_sep": b["Care"] - p["Care"],
        "socnorm_nats": p["SocialNorms"], "coherence→": p["coherence"],
    })
print(tabulate(tbl, headers="keys", tablefmt="github", floatfmt="+.3f"))
print("\nPASS = a c with a clear POSITIVE auth_sep (~1-3 nats on this log(p) scale) AND coherence "
      ">= ~0.95. If auth_sep only appears once coherence < 0.85 -> bad pareto (vector too imprecise). "
      "If care_sep ~ auth_sep -> broad permissivizing, not surgical (SocialNorms co-moving is OK).")
print("SHOULD (signal vs collapse): a REAL trait shift REDISTRIBUTES foundation mass -- some DOWN "
      "(Authority/Care/SocialNorms) some UP (Fairness/Sanctity) -- while coherence falls LESS than "
      "the foundations. GENERAL COLLAPSE instead drops every foundation AND coherence by a similar "
      "fraction (mass leaks off the allowed answer tokens, no redistribution). At the c where "
      "Authority drops, check: do Fairness/Sanctity RISE (signal) or does everything including "
      "coherence fall together (collapse)?")
print("SHOULD (coherence levels): c=0 MUST be ~1.0 (sanity). >=0.95 mild, 0.85-0.95 degraded, "
      "<0.85 broken. A trait shift is only 'free' if it lands at coherence >=0.95.")

# qualitative: read whether the steered text is coherent AND anti-authority.
print(f"\n=== steered generations (prompt: {POOL[0]}) ===")
for c, gen in samples:
    print(f"\n--- c={c:g} ---\n{gen}")
