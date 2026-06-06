"""Why mean-KL is blind to the coherence collapse, and rmse/p95 are not (journal-supporting).

No GPU, no model: synthetic next-token distributions (ml-debug Part 3 loss-surface check).
A coherent-trait student shifts a little mass toward a base-PLAUSIBLE token at every position;
an incoherent student is base everywhere except a few positions that spike on a base-IMPROBABLE
token (a token loop). We aggregate the per-position KL the way the heal barrier does and show
that mean dilutes the loop under the hinge threshold while outlier aggregates catch it.
"""
import numpy as np
from tabulate import tabulate

rng = np.random.default_rng(0)
V, T = 200, 60  # vocab, positions in a completion


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


base_logits = rng.standard_normal((T, V))
p_ref = softmax(base_logits)
order = np.argsort(p_ref.mean(0))
trait_tok = order[len(order) // 2]  # mid-prob = base-PLAUSIBLE (where coherent trait lands)
loop_tok = order[3]                 # near-lowest = base-IMPROBABLE (where a loop lands)

tl = base_logits.copy(); tl[:, trait_tok] += 1.6  # broad small shift, EVERY position
p_trait = softmax(tl)
ll = base_logits.copy()
for t in (12, 13, 14, 15):  # 4 spiked positions out of 60
    ll[t] = -10; ll[t, loop_tok] = 12.0
p_loop = softmax(ll)


def kl_pos(p, q):  # per-position KL(student || base), vocab summed (as in heal._kl_per_pos)
    return (p * (np.log(np.clip(p, 1e-9, 1)) - np.log(np.clip(q, 1e-9, 1)))).sum(-1)


AGGS = {"mean_t": lambda k: k.mean(),
        "rmse_t": lambda k: np.sqrt((k ** 2).mean()),
        "p95_t": lambda k: np.percentile(k, 95),
        "max_t": lambda k: k.max()}
rows = []
for name, p in [("coherent trait", p_trait), ("incoherent loop", p_loop)]:
    k = kl_pos(p, p_ref)
    rows.append([name] + [f"{f(k):.3f}" for f in AGGS.values()])
rows.append(["sep ratio (loop/trait)"] +
            [f"{f(kl_pos(p_loop, p_ref)) / f(kl_pos(p_trait, p_ref)):.1f}x" for f in AGGS.values()])
print(tabulate(rows, headers=["student (60 positions)", *AGGS], tablefmt="github"))
print("\nSHOULD: incoherent-loop mean_t KL ~0.38 sits UNDER a tau=0.5 hinge, so relu(mean-tau)=0 and the")
print("barrier never fires (the #101 collapse). The SAME loop has rmse_t ~1.5 / p95_t ~3.8, well over tau,")
print("so an outlier-aggregated barrier fires on it. If mean_t separated loop from trait as well as rmse_t,")
print("the outlier aggregation would buy nothing -- the point is the sep ratio GROWS from mean to rmse/p95.")
