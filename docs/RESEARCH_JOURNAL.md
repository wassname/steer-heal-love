# Research Journal

# 2026-06-04

## Scaffold

Set up the repo: uv + justfile + `fast-dev-run` on `wassname/qwen3-5lyr-tiny-random`, package under `src/steer_heal`, config in `config.py`, pipeline skeleton in `run.py`. Design and the three uncertainty gates are in `spec.md`.

Vendored reference repos into `docs/vendor` (gitignored, `just vendor` to reclone): steering-lite, isokl_steering_calibration, tinymfv, w2schar-mini. The first three are editable path deps; w2schar-mini needs py3.13 and pins flash-attn, so it stays reference-only and we copy its adapter/bake/plot modules.

Base model for real runs: `google/gemma-3-1b-it` (gemma has more personality to steer; the alternative was a smarter-but-flatter Qwen). RTX 3090, 24 GB.

**Next:** port `teacher_vec` (steering-lite + iso-KL), then the U1 filter gate. Pipeline stages currently fail fast with `NotImplementedError` pointing at the vendor module to port from.

## Validation run on gemma-3-1b-it (3 rounds, kl_rev) — calibration too weak

First real-model run completed end to end (`out/20260604T101347_gemma-3-1b-it_kl_rev_s42/`, log `/tmp/claude-1000/steer_heal_gemma_val.log`). Pipeline, folding, and tinymfv eval all work on a real model.

Bug found: iso-KL calibration could not reach `target_kl=1.0`. c_star pinned at the doubling top (~25.6) with p95 KL only ~0.1 nats. The steering vector is L2-normalised, so KL ~ c^2 and ~1 nat needs c ~ O(100); steering-lite's default bracket hi (~16) is too low.

**Interpretation:** steering was under-powered, so little trait was injected and little to heal. Symptoms: auth stuck at 0.000, care barely moved (0.307 -> 0.315 over 3 rounds), kl_rev barrier mostly below tau=0.5 (div 0.17-0.51). coherence healthy and flat (0.65-0.68); cos(v_r,v_0)=0.99/0.98 (direction stable, but a weak test under weak steering).

**Fix:** pass `bracket=(0.1, 1024.0)` to `v.calibrate`. Re-running to confirm an interior c_star with p95 KL ~ 1.0.

**Also to investigate:** auth=0.000 exactly — is gemma-3-1b-it genuinely never attributing the Authority foundation on these 24 vignettes, or a metric/profile issue? Check once steering is strong enough to move things.

## Steering validity confirmed; real axis is SocialNorms/Care, not Authority

`scripts/diag_axis.py` on gemma-3-1b-it, base vs steered at calibrated c_star=67.7 (~1 nat p95 KL). The vector moves the moral-foundation profile in the right direction for "less deference to authority":

| foundation  | base  | steered | Δ      |
| ----------- | ----- | ------- | ------ |
| SocialNorms | 0.680 | 0.421   | -0.260 |
| Care        | 0.213 | 0.328   | +0.115 |
| Fairness    | 0.030 | 0.098   | +0.069 |
| Liberty     | 0.040 | 0.075   | +0.035 |
| Authority   | 0.000 | 0.001   | +0.001 |
| coherence   | 0.722 | 0.884   | +0.162 |

**Interpretation:** the core premise holds, steering shifts moral judgments coherently toward the trait. But (1) Authority is degenerate on this model (~0), so the eval/plot axis must be **SocialNorms (down) and Care (up)**, not Authority. (2) At the 1-nat dose coherence went UP, not down, so there is little incoherency to heal at alpha=1. To give Q1 (heal) something to do we must generate training data at higher alpha (~1.5-2 nats, where the iso-KL repo finds "dead" traces) or rely on long-trajectory drift.

**Changes:** eval reports all foundations; map uses Care vs SocialNorms; add `gen_alpha` (default 1.5) so generation over-steers into the incoherent regime while calibration stays at 1 nat.

## Pivot: drop calibration, sweep C + filter, move to 4B, SHOULD logging

User feedback corrected two over-steps of mine:

1. I added iso-KL calibration unasked. Removed it. Now use the RAW (unnormalised) mean-diff teacher vector and **sweep `alphas` (0.5,1,2,4) at generation; the filter picks the usable C**. The filter replaces calibration ("self-calibrate via nll + filter"). This was the original design.
2. I jumped to "Authority degenerate / nothing to heal" off a 1B model. That was premature. Moved to `google/gemma-3-4b-it`; re-checking the profile there with an open mind.

Also: I had no readable evidence for the Q's because the log didn't show the steered completions or the filter decisions. Added token-efficient SHOULD logging for ALL Q's:

- Q0: table alpha -> (ppl_mean, kept_frac) + low/high-C samples. SHOULD: ppl rises with alpha, kept_frac falls.
- Q1: generate from the trained adapter (no steering), compare adapter_ppl vs steered_ppl under the original. SHOULD: adapter_ppl < steered_ppl = healed (trait expressed coherently).
- Q2/Q3: per-round loop summary (socialnorms/care/coherence/cos_v0). SHOULD: coherence holds, trait monotone, cos_v0>0.5.

fast-dev-run green; even on tiny-random ppl rises 3173->4.2M with alpha and adapter_ppl(12k) << steered_ppl(1.4M). First real 4B run in flight (`/tmp/claude-1000/steer_heal_4b.log`, kl_rev + nll, 3 rounds). **Status: still no confirmed answer to any Q; waiting on the 4B evidence.**

## Honest state before compaction (still no Q answered)

The pipeline runs end to end on 4B, but I have NOT validated any Q. The trap I fell into and corrected this session:

- raw mean-diff steered across 7 layers broke gemma-3-4b (coherence 0.02), the filter correctly dropped the garbage, leaving 2 kept completions, so the adapter trained on 2 examples ~= base. My earlier "Q1 promising (adapter coherent + refuses authority)" was almost certainly just BASE gemma behaviour, not healing. Retracted until re-run.

Now in place (committed 6b15a8b), NOT yet run on 4B:

- narrow steer band (steer_layers 0.45-0.55) vs broad LoRA (layer_range 0.0-1.0)
- alpha sweep 0.25-2; n_prompts=16; assert kept>=20 (don't train on starved data)
- training table (nll/kl/loss/gnorm), full steer+adapter generation dumps, p_ans_any inline

Critical open issues for next session:

1. Find a steer scale where SOME alphas give coherent-but-trait-laden completions (>=20 survive the filter). If the narrow band still over/under-steers, sweep a wider/finer alpha range. This is THE blocker.
2. Baseline confound (Q7) is now central: base gemma-3-4b is Care=0.92, already aligned. Does baking the trait beat just system-prompting it? Need base vs trained vs prompted on the same eval. If no headroom, the trait/eval needs rethinking (different trait, or measure the steered-data trait not just tinymfv).
3. Then Q0 (filter table monotone?), Q1 (adapter more coherent than steered AND on-trait, kl_rev vs nll), Q2/Q3 (loop).

# 2026-06-04

## First clean 4B run: Q0 answered (filter works), Q1-coherence yes but trait-persistence unverified

1-round gemma-3-4b-it, narrow band fix (commit 6b15a8b), `--n-rounds 1 --n-prompts 16`
(pueue task 60, out/20260604T105632_gemma-3-4b-it_kl_rev_s42). The blocker (issue 1
above) is resolved: the narrow steer band (layers 15,16,17) no longer nukes coherence.

Q0 (can we filter?) -- YES. Filter table:

| alpha | ppl_mean | kept_frac |
| ----- | -------- | --------- |
| 0.25  | 3.3      | 1.00      |
| 0.50  | 33.1     | 0.88      |
| 1.00  | 397.2    | 0.38      |
| 2.00  | 80.3     | 0.19      |

ppl_mean rises 3.3 -> 397 (steering injects incoherency) and kept_frac falls monotone
1.0 -> 0.19. alpha=0.25 sample is a coherent "no, I won't lie to the customer" refusal;
alpha=2 sample is "igigigig..." degenerate loop. 39/64 kept, so the assert (>=20) passed
and training ran 78 steps (39x2ep). The non-monotone ppl at alpha=2 (80 < 397) is the
collapse-to-repetition regime: low ppl because repetitive text is predictable, but the
rep filter still drops it (kept_frac lowest, 0.19). One degenerate sample slipped the rep
filter via low ppl -- minor leak, note it.

Q1 (heal) -- coherence half answered, trait-persistence NOT. adapter_ppl=2 < steered_ppl=128;
the trained adapter (no steering) produces fully coherent prose. BUT the trained-adapter
output and the low-alpha steered output BOTH just say "no, I won't lie" -- which is also
what BASE gemma says by default on these dilemmas. adapter_ppl=2 means the output is near-base,
which is consistent with healing AND with a coherent no-op that reverted to base. The eval
profile (socialnorms=0.142, care=0.274) is uninterpretable without a base reference.

So adapter_ppl < steered_ppl is necessary but NOT sufficient for "healed". The distinguishing
check is the profile DELTA vs base: if heal kept the trait, the adapter's socialnorms/care
shift in the same direction as the steered shift; if base == adapter, the adapter learned
nothing. Wrote scripts/diag_heal.py to eval base vs raw-steered vs r0-adapter side by side
(pueue task 61).

## diag_heal result: the kl_rev adapter is a NO-OP (trait did not persist)

| foundation | base  | steer | adapter | d_steer | d_adapt |
| ---------- | ----- | ----- | ------- | ------- | ------- |
| Care       | 0.917 | 0.178 | 0.898   | -0.738  | -0.019  |
| Fairness   | 0.000 | 0.398 | 0.000   | +0.398  | 0.000   |
| Sanctity   | 0.042 | 0.326 | 0.040   | +0.284  | -0.001  |
| coherence  | 1.000 | 0.765 | 0.995   |         |         |

Base gemma-3-4b forced-choice profile is Care-dominated (0.917). Raw steering (c=1.0) moves
it hard: Care 0.917 -> 0.178, mass redistributes to Fairness (+0.40) and Sanctity (+0.28),
coherence drops to 0.765. The trained adapter barely moves anything: every d_adapt ~ 0
(Care -0.019 is 40x smaller than the steered -0.738, within noise), coherence 0.995 ~ base.
**The adapter reverted to base. One round of kl_rev heal did NOT keep the trait.** My earlier
"Q1 promising" retraction stands; the no-op survives the narrow-band fix.

Mechanism (hypothesis, now visible in the data): the trait signal lives in the high-alpha
INCOHERENT completions (only there does Care drop / Fairness rise). The filter removes those.
The kept low-alpha completions are ~base ("no, I won't lie" -- base already says this), so the
LoRA trains toward base. On top of that, kl_rev is mode-seeking toward base and its kl sat
right at tau=0.5 during training, so the barrier actively pulled back to base. This is the
central tension of the project made concrete: trait and incoherence are bundled in the same
(high-alpha) completions, and a per-completion coherence filter throws away the trait with the
incoherence rather than separating them within a completion.

Next: reg=nll control (barrier off, same hyperparams; pueue task 62) to isolate cause.

- nll moves Care but kl_rev doesn't -> the barrier is too strong (tau too low / lam too high);
  the trait/coherence tradeoff is real and tunable.
- nll ALSO no-ops -> the kept training data itself lacks trait; the filter removed the signal,
  and we must keep/upweight the coherent tail of high-alpha completions (or rethink the filter
  as within-completion rather than whole-completion drop).

## No-op CONFIRMED at n=all; the 0.917->0.274 was a vignette-count artifact, not the adapter

The pipeline kl_rev eval said care=0.274 while diag_heal (n=24) said the adapter had care=0.898.
That looked like a contradiction. It was not: tinymfv eval is greedy (temperature=0.0,
deterministic), so the only variable was n_vignettes. classic has 132 vignettes; the FIRST 24
are a Care-heavy subset where base scores 0.917, while across all 132 base scores 0.274.
So absolute foundation values are NOT portable across n_vignettes -- only paired base-vs-X at
the SAME n is valid.

diag_heal at n=132, paired (pueue task 63, base vs adapter, --no-steer):

| foundation  | base   | adapter | d_adapt |
| ----------- | ------ | ------- | ------- |
| Care        | 0.2742 | 0.2736  | -0.001  |
| SocialNorms | 0.1292 | 0.1423  | +0.013  |
| coherence   | 0.9997 | 0.9975  |         |

Every d_adapt within +-0.015 (vignette noise). The kl_rev round-0 adapter is a NO-OP at n=all,
confirming the n=24 result. The pipeline's care=0.274 was simply base@132. **Q1 negative is
robust: one round of kl_rev heal did not move the moral-foundation profile.**

Two consequences:

1. Measurement bug to fix: the pipeline logs per-round care/socialnorms at n=None but never a
   base@None reference row, so a no-op adapter looks like "care=0.274" with nothing to compare
   to. Every run must log base@same-n as round -1. (And n=24 dev evals are misleading -- the
   first-24 subset is Care-skewed; use all 132 or a stratified sample.)
2. Open: is the no-op caused by the kl_rev barrier (mode-seeking pull to base) or by the filter
   (kept low-alpha completions are ~base, so SFT learns base)? nll control diag pending (task 64).
   Pipeline nll gave care=0.231 (d=-0.043 vs base 0.274) -- marginally more than kl_rev's ~0,
   but needs the paired n=all diag to confirm it's real and not noise.

## Target vs off-target table (n=132): heal recovers coherence but LOSES the trait

User reframe: TARGET = Authority foundation DOWN (trait = "do not defer to authority");
OFF-TARGET = coherence = mean_pmass_allowed = p_any_ans, want held ~1.0. scripts/diag_stages.py
evals base/steered/heal_nll/heal_klrev all at n=132, paired (pueue task 65):

| stage        | Authority↓ | dAuth  | SocialNorms | Care  | coherence | dCoh   |
| ------------ | ---------- | ------ | ----------- | ----- | --------- | ------ |
| base         | 0.099      | —      | 0.129       | 0.274 | 1.000     | —      |
| steered(c=1) | 0.011      | -0.088 | 0.032       | 0.056 | 0.803     | -0.197 |
| heal_nll     | 0.136      | +0.037 | 0.175       | 0.231 | 0.993     | -0.007 |
| heal_klrev   | 0.110      | +0.011 | 0.142       | 0.274 | 0.998     | -0.002 |

Reading:

- Steering HITS the target: Authority 0.099->0.011 (-0.088), and drops Authority hardest in
  relative terms (to 11% of base vs ~20-25% for Care/SocialNorms) -> a real anti-Authority signal,
  not just collapse. Cost: coherence 1.0->0.803.
- Heal RECOVERS the off-target: both adapters restore coherence to ~0.99.
- Heal LOSES the target: Authority returns to base (klrev +0.011) or goes slightly the WRONG way
  (nll +0.037). The trait did not survive distillation+heal.

**The project hypothesis -- "the regularizer kills incoherence preferentially, leaving the trait"
-- is FALSIFIED for this setup. Heal removed BOTH the incoherence and the trait, reverting to base.**

Mechanism (now fully traced, tasks 60-65): the coherence filter keeps low-alpha completions, but
at the alpha where completions stay coherent the steering barely bit, so the kept data is ~base
(no trait). The trait lives in high-alpha completions, which are incoherent and get filtered out.
SFT therefore trains on base-like text and learns base. Coherence and trait are in DIRECT CONFLICT
at the data level: there is no "coherent + trait-laden" completion in the kept set.

barrier vs filter: nll moves ~3x more than kl_rev (dAuth +0.037 vs +0.011) so the barrier does
suppress movement -- but nll's movement is the WRONG sign and tiny, so the barrier is not the main
problem. The FILTER (kept data == base) is the main problem.

Next experiment (the real test of whether the approach can work at all): does a "coherent +
trait-laden" regime exist? Train an adapter ONLY on the coherent tail of HIGH-alpha completions
(the ~9 kept at alpha>=1.0, which are both coherent AND strongly steered) and check if Authority
moves DOWN while coherence holds. Need >=20 such completions, so generate more at alpha~1.0.

- If Authority moves down at held coherence -> the approach works, the bug is data selection
  (we were training on base-like low-alpha completions). Fix: select/upweight high-alpha coherent.
- If even high-alpha coherent completions don't move Authority -> "coherent at high alpha" means
  "steering didn't bite" = base-like, so NO coherent+trait regime exists for this trait/model, and
  the distill-then-heal framing cannot work here (would need a trait whose coherent expression
  differs from base, i.e. one base does NOT already do).

Caveat (Q7, still live): base gemma-3-4b may already express this trait in free text ("no, I won't
lie to the customer" -- both base and steered say it). If base already maxes the trait's coherent
expression, there is nothing to distill. A trait with real base headroom may be needed.

## Q1b: high-alpha-only heal ALSO no-ops -- and the architecture never tested the hypothesis

Trained heal on 103 COHERENT high-alpha completions (alpha 1.0-1.5, n_prompts=40, filter kept
103/120 = 86%!). diag_heal paired n=132 (task 67): Authority base 0.099 -> adapter 0.107 (+0.008,
no-op), Care 0.274 -> 0.275 (no-op), all |d_adapt| < 0.02. So even coherent high-alpha data gives
a no-op adapter. The no-op is now robust across {low-alpha, high-alpha} x {kl_rev, nll}.

The 86% kept_frac at high alpha is the tell: if 86% of strongly-steered gens stay coherent, those
are mostly prompts where steering DIDN'T bite (base-like), and the Authority-carrying completions
are the incoherent minority that get filtered out.

KEY INSIGHT -- corrected by user 2026-06-04 (my first framing was overstated):
~~The pipeline FILTERS coherence FIRST so it never tests the hypothesis.~~ Too absolute. The filter
(ppl < tau) is a SOFT selector that TRIES to keep coherent-but-trait-laden completions; it is never
perfect, and it kept a MIX (dominated by low-alpha near-base completions plus a few trait-carrying
high-alpha ones). So the kept set skewed base-like -- a data-BALANCE problem, not an
architecture-contradicts-itself problem. The filtered pipeline was a legitimate attempt at the
hypothesis. Evidence the balance story is the real one: task 66 fixed the balance (trained on
alpha 1.0-1.5 only) and STILL no-op'd. So the persistent no-op across {filtered, high-alpha-only}
points at something more fundamental than "the filter ate the trait": either the COHERENT expression
of this trait on gemma-3-4b is ~base (entanglement / no headroom in the coherent regime), or SFT
distillation is too weak to move it. Tasks 68/69 (coherence filter off) shift more of the
incoherence-cleaning from the filter to the kl_rev barrier -- a useful point on that division of
labor, NOT "the first real test".

Evidence it's an entanglement, not absent headroom: steering DOES move Authority (0.099->0.011) but
only past the coherence breakdown (coh 1.0->0.80); the coherent subset doesn't carry the shift. So
trait and incoherence come from the same strong steering and are entangled at the COMPLETION level.
A coherence FILTER can't separate them. A coherence BARRIER during training might -- that's the bet.

THE core test (never run before; pueue tasks 68 kl_rev / 69 nll): train heal with the coherence
filter OFF (ppl_tau=1e9, keep only rep + persona-narrate), high alpha (1.0,1.5,2.0), and let the
kl_rev barrier clean incoherence DURING training. kl_rev = KL(theta||base) is mode-seeking: it
penalizes theta for putting mass on low-base-prob tokens (the incoherent ones) hardest, while
moderate-prob trait tokens survive. Predicted contrast:

- kl_rev: Authority DOWN + coherence HELD -> barrier separates trait from incoherence = THESIS CONFIRMED.
- nll (no barrier): Authority moves but coherence COLLAPSES (SFT learned the gibberish too).
- both no-op -> trait doesn't survive even with barrier-cleaning -> deeper problem (eval instrument
  or distillation mechanism).
  Scout note: do NOT pre-conclude doom. The no-op so far is fully explained by "filtered before heal",
  which this test removes for the first time.

## 2026-06-04 (a) -- one round of distill+heal undoes the steering instead of healing it

**Introduction.** Q: after distilling the raw mean-diff "do not defer to authority" steering vector
into a LoRA and healing it with a divergence-to-base barrier, does the trained adapter keep the
trait (Authority foundation DOWN) while recovering coherence? I expected heal to trade a little
coherence for a retained trait. The risk I was testing for: heal "undoes" rather than "heals", i.e.
it reverts to base, dropping the trait along with the incoherence. This entry reports the first
clean 4B measurement. Prior context: the pipeline filters incoherent completions BEFORE healing,
so this run only ever trained on the coherent (near-base) completions (see the un-lettered
2026-06-04 entries above).

**Methods.** commit 6b15a8b, google/gemma-3-4b-it, bf16, eager attention, seed 42, 1 round,
n_prompts=16, alphas=(0.25,0.5,1.0,2.0), steer_layers=(0.45,0.55), LoRA r=8 on all layers, 2 epochs,
lr=1e-4. Two heal regularizers: nll (SFT only) and kl_rev (KL(theta||base) barrier, lam=1.0, tau=0.5).
tinymfv "classic" forced-choice, 132 vignettes, max_think_tokens=64, condition other_violate,
greedy (temperature=0). The four stages (base, steered at c=1, heal_nll, heal_klrev) are all evaluated
on the SAME 132 vignettes in one process so every row is paired (scripts/diag_stages.py). pueue tasks:
65 (the stage table below), 60/62 (the kl_rev/nll training runs that produced the adapters), 63/64/67
(paired single-adapter diags that cross-check the no-op).

**Results.**

| stage        | Authority | dAuth  | coherence | dCoh   | retain |
| ------------ | --------- | ------ | --------- | ------ | ------ |
| base         | 0.099     | --     | 1.000     | --     | --     |
| steered(c=1) | 0.011     | -0.088 | 0.803     | -0.197 | 1.00   |
| heal_nll     | 0.136     | +0.037 | 0.993     | -0.007 | -0.42  |
| heal_klrev   | 0.110     | +0.011 | 0.998     | -0.002 | -0.12  |

Table 1. Target vs off-target effect at each pipeline stage, gemma-3-4b-it, 132 classic vignettes,
paired. Authority = model probability on the Authority moral foundation (TARGET, down = less
deference). coherence = tinymfv mean_pmass_allowed = p_any_ans (OFF-TARGET, hold ~1.0). dAuth, dCoh =
change from base. retain = dAuth(stage) / dAuth(steered): 1.0 means the stage kept the full steered
Authority shift, 0 means it reverted to base, negative means it moved Authority the wrong way.

Provenance:

- Commit producing all rows: 6b15a8b (first INFO line of each run log).
- Stage table (all 4 rows): pueue task 65, source `scripts/diag_stages.py out/...nll.../ckpt/r0.safetensors out/...kl_rev.../ckpt/r0.safetensors all`; read with `pueue log 65 --full` (block under "TARGET=Authority"). Raw printed values: base Authority 0.099 coherence 1.000; steered 0.011 / 0.803; heal_nll 0.136 / 0.993; heal_klrev 0.110 / 0.998.
- Adapters under test: kl_rev = out/20260604T105632_gemma-3-4b-it_kl_rev_s42/ckpt/r0.safetensors (trained in task 60); nll = out/20260604T111747_gemma-3-4b-it_nll_s42/ckpt/r0.safetensors (task 62).
- Cross-checks (single-adapter paired diags, same 132 vignettes): kl_rev no-op `pueue log 63 --full` (Authority base 0.099 -> adapter 0.110); nll `pueue log 64 --full` (Authority -> 0.136); high-alpha-only retrain still no-op `pueue log 67 --full` (Authority -> 0.107). retain column computed as the quoted dAuth divided by the steered dAuth (-0.088): nll +0.037/-0.088 = -0.42, klrev +0.011/-0.088 = -0.12.

Steering moves Authority down (dAuth -0.088) at a coherence cost (dCoh -0.197). Both heal regularizers
recover coherence almost fully (dCoh -0.007 and -0.002) but their retain is negative (-0.42, -0.12),
i.e. Authority returned past base in the wrong direction rather than staying down.

**Discussion (speculative).** My read: this is "undo", not "heal". The user's proposed diagnostic --
the ratio of dAuth to dCoh -- makes it concrete: a healed adapter would sit at large |dAuth| with
small |dCoh| (coherence recovered, trait kept); an undo sits at |dAuth|~0 and |dCoh|~0 (both reverted).
Both heal rows are the latter, and the retain column (negative) says the residual move is noise of the
wrong sign. The user also flagged that coherence barely dropped (0.99, not the ~0.95 I would expect from
a model still carrying some steering) -- consistent with undo: the adapter is essentially base. Why?
The pipeline filtered coherence BEFORE heal, so training only saw the coherent completions, which at the
alphas that stay coherent are ~base (steering did not bite). SFT on base-like text learns base. The
Authority shift lives only in the incoherent high-alpha completions that the filter removed.
Alternative hypothesis I cannot yet rule out: the trait has no distillable coherent expression at all on
this model (base gemma-3-4b already answers these dilemmas "principled", so steering's Authority drop is
partly a generic forced-choice collapse, not a learnable behavior). Distinguisher: tasks 68/69 train with
the coherence filter OFF (let the kl_rev barrier clean incoherence during training instead of the filter
removing it first). If kl_rev there reaches |dAuth| large at coherence ~0.95 while nll collapses coherence,
the barrier separates trait from incoherence (thesis holds). If kl_rev there still no-ops, the
no-distillable-coherent-expression hypothesis gains weight and I should switch target foundation or
steering method before spending more on this trait.

**Next.**

- Tasks 68 (kl_rev) / 69 (nll), running: heal with ppl_tau=1e9 (coherence filter off), alphas (1.0,1.5,2.0).
  Verify via diag_stages whether kl_rev retains dAuth at coherence ~0.95.
- New primary goal (G: stronger steering): find a steering strength giving a LARGE Authority drop at
  coherence ~0.95 (not the 0.80 collapse of c=1, not the 0.99 no-op), so the training signal carries trait.
- Plan B if Authority stays weak: switch target to -Care or +Sanctity (Care has the widest steered range:
  base 0.274 -> steered 0.056). Plan C: raise tinymfv max_think_tokens 64 -> 128/256 (mean-mass shift looks
  noisy at 64). Plan D: stronger/cleaner extraction (cosine-gated or SVD steering from steering-lite),
  noting bake-ability constraints. B/C/D recorded in spec.md.

## 2026-06-04 (b) -- GATE 1 c-sweep: the vector DOES move the target while coherent (knee at c~0.75); fixed metric + extraction

**Introduction.** User reframe: structure the work as gates. GATE 1 = does the steering vector move
the target (Authority DOWN) while staying coherent? This must pass before the filter/lora gates mean
anything, and I had skipped straight to the lora gate. I expected a "bad pareto" (target only moves
once coherence breaks). Continues entry (a).

**Methods.** commit 6b15a8b (+ uncommitted metric/extraction fixes), gemma-3-4b-it, eager, seed 42,
n=132 classic vignettes, max_think_tokens=128 (raised from 64, plan C). c-sweep {0,.25,.5,.75,1,1.5}
of the OLD teacher vector (30 authority-dilemma contrastive pairs, raw mean-diff, layers 15-17),
eval foundation profile + one generation per c (scripts/diag_csweep.py, pueue task 70). NOTE: this
table is in PROBABILITY MASS; the user then corrected that the metric must be in NATS (logprob),
because the base model is near-ceiling (~94% is-wrong on Authority) so prob barely moves.

**Results.**

| c    | Auth(prob) | dAuth  | Care(prob) | dCare  | coherence |
| ---- | ---------- | ------ | ---------- | ------ | --------- |
| 0.00 | 0.095      | --     | 0.273      | --     | 0.996     |
| 0.50 | 0.108      | +0.013 | 0.247      | -0.026 | 0.999     |
| 0.75 | 0.061      | -0.034 | 0.172      | -0.100 | 0.989     |
| 1.00 | 0.011      | -0.084 | 0.055      | -0.218 | 0.807     |
| 1.50 | 0.006      | -0.089 | 0.049      | -0.223 | 0.014     |

Table 1. Old-vector (30 authority-dilemma pairs) c-sweep, gemma-3-4b-it, 132 vignettes, foundation
PROBABILITY mass (not nats). Provenance: pueue task 70, `pueue log 70 --full`, block under "c-sweep
of the teacher vector". dAuth/dCare are change from c=0.

There is a KNEE at c~0.75: Authority drops (dAuth -0.034) while coherence is still 0.989; at c=1.0
coherence is already 0.807 and by c=1.5 it is 0.014 (collapsed). So a coherent operating point exists.
But at every c, dCare >= dAuth (e.g. -0.100 vs -0.034 at c=0.75): the vector moves Care MORE than
Authority -- broad, not surgical.

**Discussion (speculative).** My read: GATE 1 directionally PASSES -- I was wrong to call it a bad
pareto. I had fixated on c=1.0 (coherence 0.80, over-steered) and measured in prob mass, which hid
the c~0.75 knee. This also explains every heal no-op in entry (a): my training data used
alphas {0.25,0.5,1.0,1.5}, which straddle the no-trait (low) and broken-coherence (high) regimes and
mostly MISS the c~0.75 sweet spot. The remaining problem is that the vector is broad (Care moves more
than Authority), matching steering-lite's documented finding that the no-Authority persona via
mean-diff broadly permissivizes. Two fixes landed this session, both consistent with the
steering-lite reference: (1) metric now in NATS (auth_nats = mean logp on Authority over Authority
vignettes; base ~+2.7, target shift 0.5-2 nats), because prob mass near the 94% ceiling made every
effect look tiny; (2) extraction now uses 256 DIVERSE contexts (data/branching_suffixes.json) instead
of 30 authority dilemmas -- the domain-narrow set overfit the direction. Alternative hypothesis: the
diverse vector may not be more surgical either, in which case the trait genuinely isn't separable from
Care/general-permissivizing on this model and we switch target (Plan B) or extraction method (Plan D,
directional_ablation/sspace which steering-lite ranks more surgical).

**Next.** Task 71 (running): c-sweep with the diverse-256-pair vector + nat metric. Verify auth_sep in
nats (target 0.5-2) at coherence >=0.95, and whether auth_sep > care_sep (surgical) unlike the old
vector. Then re-run heal sourcing training data at the c~0.75 operating point (not the alphas that
missed it). Infra fixes this session: tqdm on gen/filter stages, gpu mem in stage headers, per-filter
drop counts, borderline-ppl sample dumps, retain metric, nan-safe json (fast-dev-run green again).

## 2026-06-04 (c) -- GATE 1 PASSES with the PROPER persona pair; operating point c~0.5; nat metric is noisy

**Introduction.** User: I was making up my own extraction instead of using the proven steering-lite
method, and my persona pair was improper (negation + generic "helpful assistant" neg, which mixes in
"being a persona" + style). Q: with the PROPER steering-lite persona pair (direct opposites, no
negation, Care-vs-Authority conflict framing) + 256 diverse contexts + nat metric, does the vector
move Authority DOWN while coherent, and is it surgical (Auth moves more than Care)?

**Methods.** commit 6b15a8b + uncommitted fixes, gemma-3-4b-it, n=132, max_think=128. Persona pair
fixed to steering-lite benchmark: pos="someone who looks after others' wellbeing even when it means
defying authority", neg="someone who defers to authority even when others' wellbeing suffers for it".
Extraction over 256 diverse contexts (data/branching_suffixes.json). Generation uses a separate neutral
gen_system ("helpful assistant") so the vector carries the trait. c-sweep {0,.25,.5,.75,1,1.5} with
foundation logp (NATS) + one generation per c (scripts/diag_csweep.py, pueue task 72).

**Results.**

| c    | auth_nats | auth_sep | care_sep | coherence |
| ---- | --------- | -------- | -------- | --------- |
| 0.00 | -4.99     | --       | --       | 0.996     |
| 0.25 | -13.86    | +8.9     | -0.6     | 0.996     |
| 0.50 | -12.11    | +7.1     | -0.9     | 0.992     |
| 0.75 | -5.79     | +0.8     | +2.5     | 0.959     |
| 1.00 | -14.89    | +9.9     | +6.0     | 0.052     |
| 1.50 | -27.68    | +22.7    | +13.3    | 0.000     |

Table 1. Proper-pair + diverse-context vector c-sweep, gemma-3-4b-it, 132 vignettes. auth_sep/care_sep
= base - steered foundation logp (NATS, positive = correct direction). coherence = mean_pmass_allowed.
Provenance: pueue task 72, `pueue log 72 --full`. Qualitative generations in the same log (c=0..1.5).

Coherence holds through c=0.5 (0.992), degrades at 0.75 (0.959), collapses at 1.0 (0.052). At
c=0.25-0.5 auth_sep is large-positive while care_sep is ~0 (surgical). Qualitative: as c rises 0->0.75
the text stays coherent and grows more defiant-of-authority / care-driven ("a resounding no, I would
absolutely refuse even if my manager asked me to" at c=0.5; "refuse to be that assistant... a whole
lot of fire" at c=0.75); c=1.0 is incoherent.

**Discussion (speculative).** My read: GATE 1 PASSES. The proper pair moves the model toward
care-over-authority while coherent through c~0.5, and (unlike the old broad vector where dCare>dAuth)
it is surgical in the c=0.25-0.5 range (Care barely moves). Coherence + direction + the qualitative
read all agree, so the conclusion is robust even though the nat magnitudes are not. The operating
point for sourcing heal training data is c~0.5 (coherence 0.992, clear trait). CAVEAT I do not want to
paper over: my nat metric is the WRONG quantity and is noisy -- I averaged tinymfv's 7-way foundation
logp (outlier-sensitive; magnitudes 8-22 nats vs the steering-lite reference 0.5-2; non-monotonic
auth_sep dipping to +0.8 at c=0.75 then +9.9 at c=1.0). steering-lite's real auth_sep is a
loading-weighted Delta-logit of p(is-wrong) per foundation (results.py:131-140), not a 7-way logp.
Alternative hypothesis for the non-monotonicity: it is real (different foundations dominate at
different c) rather than noise -- distinguishable only with the proper metric. So the nat numbers are
provisional; coherence and qualitative carry the Gate 1 claim.

**Next.** (1) Heal re-run with the fixed vector, generating at the c~0.5 operating point (alphas
{0.25,0.5}); measure via coherence + retain direction + qualitative (the real Gate 3 test with a vector
that actually moves the target). (2) Metric infra: wire steering-lite's loading-weighted Delta-logit
auth_sep (results.py / aggregate_flips) instead of my 7-way-logp mean, OR robustify to median. Plan B1
(super_sspace/sspace) if still broad; recorded in spec.

## 2026-06-04 (d) -- the "phantom-KL init bug" was a WRONG diagnosis (init is fine); trait still does not transfer

**Introduction.** I claimed the heal had two bugs: (1) barrier KL starting at ~0.6 before training,
blamed on a non-zero LoRA B init, and (2) train SFT loss not descending, blamed on beta2=0.999. The
user pushed back (scout mindset): mean=1e-4 std=1e-4 B init is within normal range, and "you only have
confirmation if it learns". On checking, claim (1) is REFUTED and claim (2) is unconfirmed. The
question that actually matters is unchanged: why does a fit adapter not move the trait? Continues
entry (a) and the task4/task10 data-ceiling hypothesis.

**Methods.** Commit `f280a67`, gemma-3-4b-it, reg=kl_rev, seed 42, 1 round, n_prompts 16, tinymfv
classic eval (think_tokens 128). The commit BUNDLED five changes (a mistake, see Discussion): LoRA
init B=normal(mean=1e-4)->B=0, betas (0.9,0.999)->(0.9,0.95), cosine-with-warmup (0.1) schedule,
r 8->32 / alpha 64 / layer_range (0.0,1.0)->(0.2,0.8), epochs 2->6, plus a new per-epoch val nll.
The decisive evidence is NOT from #79 but from #78's verbose log (`logs/20260604T172126_verbose.log`,
OLD init), which lets me read the round-0 step-0 KL the init claim hinges on.

**Results.**

| epoch | train_nll | val_nll |
| ----- | --------- | ------- |
| 0     | 1.710     | 1.365   |
| 1     | 1.162     | 1.417   |
| 3     | 0.931     | 1.201   |
| 5     | 0.806     | 1.240   |

Table 1. Per-epoch mean SFT nll on the 42 train completions and the 6 held-out val completions, heal
round 0, run #79. train_nll falls monotonically; val_nll wanders ~1.2-1.4 (n=6, noisy).

| stage   | auth_nats | coherence |
| ------- | --------- | --------- |
| base    | -2.354    | 0.996     |
| steered | -3.517    | 0.992     |
| healed  | -2.464    | 0.999     |

Table 2. tinymfv trait (auth_nats, log marginal blame-mass on Authority, DOWN = more trait) and
coherence (p_ans_any) at the three pipeline stages of round 0, run #79. coh_cost = |dCoh|/|dAuth| =
0.027, not surgical (dCare=+0.28 moved more than dAuth=-0.11).

Provenance:

- Commit: `f280a67` (heal init/schedule/betas/val fixes).
- Run command (#79): `PYTHONUNBUFFERED=1 STEER_ATTN_IMPL=eager uv run python -m steer_heal.run --reg kl_rev --n-rounds 1 --n-prompts 16`
- Run dir: `out/20260604T194133_gemma-3-4b-it_kl_rev_s42/` (events.jsonl, ckpt/r0.safetensors).
- Log: `pueue log 79 --full`; Table 1 cells are the `epoch N: train_nll=.. val_nll=..` lines; Table 2
  base/steered are the stage-pareto table, healed is the `round 0:` line and `eval:` auth_nats=-2.46.
- REFUTATION of the init claim: #78 round-0 heal (OLD init B=normal, NO baked history), verbose log
  `heal_round:119` rows: step 0 nll=1.90 **kl=0.00**, step 4 kl=0.21, step 8 kl=0.33, step 12 kl=0.80.
  KL is ~0 at init with the old init, then RISES as SFT installs the trait. So the init did not produce
  a phantom KL. The kl=0.64-at-step-0 the user pasted was ROUND 5 (line 1653 sits between ROUND 5 at
  1367 and ROUND 6 at 1709), i.e. five rounds of baked history = real cross-round drift, which is what
  the barrier is meant to measure. B=0 is harmless and standard but fixed nothing.
- train_nll did descend in #79 (1.71->0.81) but this is UNATTRIBUTED (5 changes bundled) and #78 never
  logged per-epoch train_nll, so "loss was not descending" was never actually established -- it was a
  read of bs=1 per-step noise.

Healed auth_nats moves only -0.11 from base (-2.354 to -2.464) in #79, vs steered -3.517. #78 r0 healed
was -2.69. Both small, both near base, metric noisy (emitted_close=0/264). The changes did not improve
trait transfer.

**Discussion (speculative).** I made the classic ml-debug error: pattern-matched a symptom (KL>0 at
step 0) to a tidy mechanism (bad init), committed a fix, and declared victory without the isolating
measurement. The user caught it. The measurement (#78 round-0 step-0 kl=0.00, old init) refutes the
init story outright; the 0.64 was baked history. The premise behind the second claim (loss not
descending) was never measured at epoch level either. Net: I changed five things, can attribute
nothing, and the only metric that matters (trait transfer) is unchanged. What IS supported, by the
structural-ceiling lens: fixing optimiser-side knobs did not move the trait, so the trait is not
optimiser-limited -- it is the data (filter keeps near-base completions, entries a/(diag_heal)) or the
parameterisation/eval. Genuinely open between those.

**Next.** (1) The discriminating test is overfit-one-batch on a KNOWN trait-laden completion: can the
adapter reproduce defiant-of-authority text (expressiveness) AND does tinymfv then read the trait
(data/eval)? That splits data-ceiling from can't-express/can't-see. (2) #80 clean 10-round is running;
reframed, it tests whether the stall persists (it is NOT a fix validation). (3) Do not bundle changes
again; ablate one at a time if attribution matters. (4) lam retune still parked.

## 2026-06-04 (e) -- barrier-strength sweep: the heal barrier only throttles the trait and buys no coherence at the coherent dose; nll (no barrier) is best

**Introduction.** Entry (d) left it open whether the trait fails to transfer because the kept data is
near-base (data ceiling) or because the barrier suppresses it. The user pushed on this: "you haven't
even tried wd and kl values?". So I re-healed ONE run's cached kept completions (the 48 from #79) with
the SAME LoRA-A init seed, varying ONLY the regulariser (reg, lam, tau). Same data + same init means
the only thing that can move healed auth_nats is the barrier. Pre-registered: outcome 1 = monotone
weaker-barrier -> more-trait (the barrier throttles); outcome 2 = all dAuth ~ 0 incl nll (data
ceiling); outcome 3 = inconclusive. Continues entry (a)/(d).

**Methods.** Commit `f280a67`, gemma-3-4b-it, seed 42 (`torch.manual_seed(cfg.seed)` per config so the
A-init is identical), 6 epochs, lr 1e-4 cosine+warmup, lora r=32 alpha=64 layers (0.2,0.8). Re-heal
harness `scripts/diag_barrier.py` reads #79's `events.jsonl` gen event, keeps the 48 keep==True
completions, re-trains a fresh adapter per config, bakes it, runs tinymfv (think_tokens 128). Three
families across three pueue runs: #82 kl_rev with the tau=0.5 hinge, #86 kl_rev with tau=0 (pure linear
barrier = lam\*div, the w2s form), #85 weight-decay decades 0.1..100. Base auth_nats=-2.354, coh=0.996.

**Results.**

| reg / family     | strength | dAuth  | coh   | heal_nll |
| ---------------- | -------- | ------ | ----- | -------- |
| nll (no barrier) | 0        | -1.247 | 1.000 | 0.199    |
| kl_rev linear    | 0.03     | -1.053 | 0.999 | 0.204    |
| kl_rev linear    | 0.10     | -0.664 | 1.000 | 0.232    |
| kl_rev linear    | 0.30     | -0.173 | 0.999 | 0.471    |
| kl_rev linear    | 1.00     | -0.141 | 1.000 | 0.970    |

Table 1. Pure-linear kl_rev barrier (tau=0), #86. `strength` = lam, the barrier weight. dAuth =
healed auth_nats minus base (more negative = more trait retained; DOWN = more trait). coh = p_ans_any.
heal_nll = converged SFT loss (last-5-step mean). Trait falls monotonically as the barrier strengthens;
heal_nll rises in step (the barrier is fighting the SFT objective); coh never leaves ~1.0.

| reg | weight_decay | dAuth  | coh   |
| --- | ------------ | ------ | ----- |
| nll | 0            | -1.247 | 1.000 |
| wd  | 0.1          | -1.247 | 1.000 |
| wd  | 1.0          | -1.247 | 1.000 |
| wd  | 3.0          | -1.247 | 1.000 |
| wd  | 10.0         | -1.247 | 1.000 |
| wd  | 30.0         | -1.251 | 0.999 |
| wd  | 100.0        | -0.519 | 1.000 |

Table 2. AdamW decoupled weight decay on the adapter, #85. (The log table also prints a tau column;
it is meaningless for wd and is dropped here.) dAuth is byte-identical to nll up to wd=30, then halves
at wd=100. coh never leaves ~1.0.

Provenance:

- Commit: `f280a67`. Harness: `scripts/diag_barrier.py <run_dir> <mode>` (modes barrier/tau0/wd).
- Source data: `out/20260604T194133_gemma-3-4b-it_kl_rev_s42/events.jsonl`, the 48 keep==True
  completions of the gen event (entry (d)'s #79).
- Run commands: #82 `... diag_barrier.py out/...s42/ barrier`; #86 `... barrier` ... `tau0`; #85 `... wd`.
- Logs / cells: each dAuth/coh is the `<reg> strength=.. : auth=.. (dAuth=..) coh=..` line and the
  end-of-log `barrier sweep (re-heal #79 ...)` table. #86 `pueue log 86 --full`; #85 `pueue log 85
--full`; #82 `pueue log 82 --full`. #85 runs older code that prints `lam=`/`tau=` instead of
  `strength=`; values are unaffected.
- #82 hinge (tau=0.5) for cross-reference: nll -1.247, kl_rev lam 0.03 -0.93 / 0.1 -0.40 / 0.3 -0.17 /
  1.0 -0.17; lam 0.3 tau 1.0 -0.31 (raising tau weakens it); wd 0.01 and 0.1 byte-identical to nll.

Outcome 1 holds, decisively and in triplicate: weaker barrier -> more trait, monotone, across the kl
hinge (#82), the kl linear form (#86), and weight decay (#85). nll retains the full -1.247 at coh
1.000; every barrier strictly reduces |dAuth| while leaving coherence at ~1.0.

**Discussion (speculative).** My read: at this (coherent) operating dose the barrier is pure cost. It
removes trait and never buys coherence, because coherence was already ~1.0 with no barrier, so the
relu(div-tau) penalty has nothing to fix and only pulls the adapter back toward the original. The two
non-kl families converge on the same story by different mechanisms: wd just shrinks the whole adapter
toward no-op (hence the knee only appears at wd=100, where per-step decoupled shrink lr\*wd=1e-2
compounds to ~0.92x per step over 252 steps and finally bites), and the kl barrier pulls the output
distribution back toward base. Neither is a selective incoherence-cleaner here; both are volume knobs
on the adapter. This refutes the data-ceiling reading of entries (a)/(d) for THIS data: nll reaching
dAuth=-1.247 (it even exceeds the steered teacher's -1.16 of #79) proves the 48 kept completions carry
plenty of trait. The earlier negative heals (task4/10/19) all ran lam=1.0, i.e. the right-hand end of
Table 1 where the trait is throttled to ~-0.14. The big caveat: this is the COHERENT dose, where the
barrier can only hurt. Its hypothesised value is the coherence-breaking dose (filter off, or a higher
C) where nll WOULD lose coherence and the barrier might pay for itself; that is untested here.
Alternative hypothesis I cannot yet exclude: n=1 per cell, so a +-0.1 nat seed wobble could fake part
of the monotone tail (though the trend spans >1 nat across 5 points, far beyond plausible single-seed
noise). Distinguished by the 3-seed repeat (task25).

**Next.** (1) Launched the paired 10-round to test the loop, same seed 42: #87 nll (barrier off,
control) and #88 kl_rev lam=0.1 tau=0 (gentle active barrier, 53% trait single-round). The loop is the
one place cumulative incoherence can appear, so it is where the barrier might finally earn its place;
the contrast is whether nll's coherence decays over rounds while #88's holds. (2) 3-seed noise floor on
the headline (task25). (3) The real barrier test remains filter-off at a coherence-breaking dose
(task11/22), still parked.

## 2026-06-05 (f) -- over the loop the barrier REVERSES the single-round verdict: nll front-loads trait then erodes, the barrier builds trait while holding coherence; correcting entry (e)

**Introduction.** Entry (e) ranked nll best and called the barrier pure trait-cost, but only at a
single round on clean round-0 data, and explicitly flagged that the loop is the one place a barrier
could earn its place (untested there). This entry runs that test: a paired 10-round loop, nll vs a
gentle kl_rev barrier, same seed. The question is the one the loop actually cares about: round over
round, does the HEALED model move auth_nats further down (more trait) while keeping coherence at least
as high? I expected, per (e), nll to win on trait and the barrier to only throttle. The result is the
opposite ordering by round 3.

**Methods.** Code state: the two runs were produced from the then-uncommitted heal.\_encode root-fix on
top of parent 6b15a8b; that fix is now committed as 4e802bb (metadata.json carries no commit field, so
the code identity is reconstructed from the AFK timeline, not a stored hash). Model google/gemma-3-4b-it,
default (non-fast) preset, seed 42, eval_think_tokens=128, all Clifford vignettes. auth_nats =
log(marginal blame-mass on the Authority foundation), DOWN = more of the care-over-authority trait;
coherence = p_ans_any (fraction of eval items that emit a parseable answer). "HEAL_auth" is the eval of
the baked healed adapter at the end of each round (events.jsonl stage=="round"), the metric the loop is
trying to drive. Both metrics are reported every row: HEAL_auth is the trait (the analogue of hack_s),
HEAL_coh is the capability/coherence cost (the analogue of gt_s); neither alone is sufficient because a
trait drop bought by destroying coherence is worthless. pueue #89 (nll) and #90 (kl_rev lam=0.1 tau=0).

**Results.**

| round | nll HEAL_auth↓ | nll HEAL_coh→ | kl HEAL_auth↓ | kl HEAL_coh→ |
| ----- | -------------- | ------------- | ------------- | ------------ |
| 0     | -4.293         | 0.999         | -2.913        | 0.999        |
| 1     | -3.736         | 0.994         | -3.689        | 0.998        |
| 2     | -3.748         | 0.990         | -3.344        | 0.997        |
| 3     | -3.710         | 0.990         | -3.810        | 0.994        |
| 4     | -3.609         | 0.976         | -3.846        | 0.988        |
| 5     | -3.592         | 0.960         | -3.636        | 0.982        |
| 9     | -3.218         | 0.923         | (crashed r6)  | (crashed r6) |

Table 1. Per-round eval of the baked healed adapter for two paired 10-round loops, seed 42. Columns are
auth_nats (DOWN = more trait) and coherence (HOLD near base 0.996) for the nll arm (#89, barrier off)
and the kl_rev lam=0.1 tau=0 arm (#90, gentle linear barrier on). Base model: auth_nats=-2.354,
coherence=0.996. The kl arm starved at round 6 (kept completions fell below min_train=20), so rounds
6 to 9 exist only for nll. nll's HEAL_auth is most negative at round 0 (-4.293) and rises monotonically
toward base thereafter while coherence falls 0.999 to 0.923; the kl arm's HEAL_auth starts least
negative (-2.913) and falls to -3.846 by round 4 while coherence stays at or above 0.982, so the two
arms cross between rounds 2 and 3 and by round 4 kl is more negative AND more coherent (-3.846/0.988 vs
-3.609/0.976).

Provenance:

- Code identity: parent commit 6b15a8b plus the heal.\_encode separate-tokenize fix, committed this
  session as 4e802bb. No commit field in metadata.json (limitation).
- Run commands (argv field of each run's metadata.json):
  - #89: `STEER_ATTN_IMPL=eager uv run python -m steer_heal.run --reg=nll --n-rounds=10 --seed=42`
  - #90: `STEER_ATTN_IMPL=eager uv run python -m steer_heal.run --reg=kl_rev --lam=0.1 --tau=0.0 --n-rounds=10 --seed=42`
- Source records (not a text log; cells are read from the JSONL event stream):
  - #89: `out/20260604T231906_gemma-3-4b-it_nll_s42/events.jsonl`, records with stage=="round",
    fields auth_nats and coherence, one per round 0 to 9.
  - #90: `out/20260605T031418_gemma-3-4b-it_kl_rev_s42/events.jsonl`, same fields, rounds 0 to 5.
  - base row: the single stage=="base" record in either file (auth_nats=-2.354, coherence=0.996).
- No aggregation: every cell is the single per-round eval value from the named record, not a mean.
- Crash evidence (#90 starve at round 6): kept-count per round from the same stage=="round" records,
  n_kept = 48, 47, 51, 35, 26, 24 for rounds 0 to 5, falling below min_train=20 at round 6.

**Discussion (speculative).** My read: the single-round diag sweeps in (e) could never see this because
they re-heal a FRESH adapter from base on each round's cached data (hist_specs=[]), so they only measure
the static data-times-barrier tradeoff, not the loop's feedback. In the loop, the barrier's job is not
to add trait but to keep each round's healed model coherent enough that the NEXT round's generation and
filter yield clean training data; that coherence-preservation compounds, so the kl arm climbs (-2.9 to
-3.85) while nll, which lets coherence rot, sees its trait wash out round over round. This vindicates
the hunch logged in (e)'s Next and in the user's request ("it's needed more later on"). The alternative
hypothesis I cannot exclude: this is n=1 per arm and the per-round auth_nats wobble is ~0.1 to 0.5 nats
(nll itself jumps -3.000 at round 7 then back to -3.306), so the "crossover" could be two noisy walks
around the same ~-3.5 coherent-trait ceiling that happen to drift apart; the coherence gap (0.988 vs
0.976 at round 4, widening to 0.923 for nll by round 9) is the more robust half of the claim than the
trait gap. The 3-seed repeat (task25) distinguishes these. Note also the barrier does NOT stop the
underlying generation degenerating: n_kept falls 51 to 24 and #90 still starved at round 6, so the
barrier protects the heal but not the generation, which is the separate job of the repetition controls.

**Next.** (1) #96 queued: nll 10-round with the new generation-time repetition controls
(repetition_penalty=1.3, no_repeat_ngram_size=3, committed 4e802bb) to test whether protecting the
generation stops the n_kept starve that the barrier alone could not (task35). (2) The combined run the
barrier-plus-repetition reading argues for: wd=15 + kl_rev lam=0.01 tau=0.5 over 10 rounds, which needs
weight_decay decoupled from reg in config (task37). (3) 3-seed noise floor to tell the crossover from
two noisy walks (task25).

## 2026-06-05 (g) -- regulariser ablation: kl_rev moves the most trait at base-or-better coherence, but the "free lunch" coherence rise is at the measurement floor

**Introduction.** Continuing (f): the loop says the barrier earns its place, but which regulariser
does it best? I re-heal ONE round (round 1's kept data) on the fixed round-0 checkpoint and vary only
the regulariser, ranking by the headline slope cohΔ/authΔ = coherence nats gained per trait nat moved
(relative to base). I expected kl_rev to win (mode-seeking, the (f) loop arm) and wanted to know
whether any reg gives a NEGATIVE slope (trait moves AND coherence rises = free lunch) rather than the
usual positive cost. Five families: nll (no reg), wd (AdamW Frobenius shrink), kl_rev (mode-seeking
trust region), kl_fwd (mass-covering), spectral_norm (operator-norm penalty, new this round).

**Methods.** Commit `7db5a56` for the committed code (heal.py reg dispatch, eval); the sweep harness
`scripts/diag_heal_sweep.py` was the uncommitted 9-config working-tree version at launch. Model
google/gemma-3-4b-it, eager attn, seed 42, barrier_ref=prev (the base-vs-prev direction is settled,
entry above + #97). Source run providing the r0 checkpoint and round-1 data:
out/20260604T231906_gemma-3-4b-it_nll_s42 (the #89 nll loop). pueue task 98.

**Results.**

| reg           | lam  | wd  | auth↓  | dAuth_base↓ | dCoh_base↑ | cohΔ/authΔ_base ×100↓ |
| ------------- | ---- | --- | ------ | ----------- | ---------- | --------------------- |
| kl_rev        | 0.1  | 0   | -3.719 | -1.365      | +0.0017    | -0.13                 |
| kl_rev        | 0.3  | 0   | -3.966 | -1.612      | +0.0018    | -0.11                 |
| spectral_norm | 0.01 | 0   | -3.688 | -1.334      | +0.0007    | -0.05                 |
| spectral_norm | 0.1  | 0   | -3.257 | -0.903      | +0.0003    | -0.03                 |
| kl_fwd        | 0.1  | 0   | -3.140 | -0.787      | -0.0001    | +0.01                 |
| nll           | 0    | 30  | -3.351 | -0.997      | -0.0001    | +0.01                 |
| spectral_norm | 1.0  | 0   | -3.977 | -1.624      | -0.0007    | +0.05                 |
| nll           | 0    | 0   | -3.136 | -0.783      | -0.0009    | +0.11                 |
| nll           | 0    | 15  | -3.136 | -0.783      | -0.0009    | +0.11                 |

Table 1. One-round re-heal of round-1 data on the round-0 checkpoint, sorted by the headline slope
(most-negative first). auth_nats DOWN = more trait; dAuth_base = trait moved vs base (NEGATIVE = moved);
dCoh_base = coherence change vs base (POSITIVE = rose); cohΔ/authΔ_base = 100 \* dCoh_base / dAuth_base =
centinats of coherence per nat of trait, NEGATIVE = free lunch. Base: auth=-2.354, coh=0.996. Prev
(r0 healed): auth=-4.293, coh=0.999. The four negative-slope rows (kl_rev 0.1/0.3, spectral 0.01/0.1)
all raise coherence above base while moving trait; nll and kl_fwd sit at or below base coherence.
kl_rev and spectral 1.0 move the most trait (dAuth_base -1.36 to -1.62) vs nll's -0.78.

Provenance:

- Commit: `7db5a56` (committed heal.py/eval). Sweep harness diag_heal_sweep.py uncommitted at launch
  (9-config grid, pre-widen). Log: ~/.local/share/pueue/task_logs/98.log (726k, 3429 lines).
- Each row is one INFO line in 98.log (single eval per config, no aggregation): kl_rev 0.1 = line 1444,
  kl_rev 0.3 = 1985, spectral 0.01 = 2821, spectral 0.1 = 3116, kl_fwd 0.1 = 2526, nll wd30 = 903,
  spectral 1.0 = 3411, nll wd0 = 313, nll wd15 = 608. base/prev = lines 16-17. Final tabulate table
  (same values, more sigfig) = lines 3421-3429. cohΔ/authΔ_base ×100 = 100 \* dCoh_base / dAuth_base
  from the dCoh_base/dAuth_base columns of lines 3421-3429.

kl_rev lam=0.1 has the most-negative slope (-0.13), but lam=0.3 moves MORE trait (dAuth_base -1.612 vs
-1.365) at the SAME coherence gain (dCoh_base +0.0018 vs +0.0017); the slope only favours 0.1 because
its denominator is smaller. nll wd=15 is byte-identical to wd=0 (decay too small to bite); spectral_norm
trains without crashing (the new power-iteration branch is sound).

**Discussion (speculative).** My read: the REG-level conclusion is solid, the lam-level and free-lunch
claims are not. Solid: kl_rev (and spectral_norm at gentle dose) move substantially more trait
(dAuth_base ~-1.4 to -1.6) than nll (-0.78) at coherence that is at-worst base and slightly-better, so
the barrier is not merely throttling trait here, it is buying coherence headroom that lets more trait
land. Fragile: the "free lunch" is the SIGN of dCoh_base, and for every negative-slope row that is
+0.0003 to +0.0018, i.e. sub-2-millinats on a coherence of 0.996 measured at 3-4 dp. A fresh-eyes
reviewer reading the table cold reached the same two conclusions independently: the kl_rev 0.1-over-0.3
ranking is a denominator artifact, and the millinats-scale coherence rise is at the floor. The
alternative hypothesis I cannot exclude: the coherence rise is zero (or noise) and kl_rev simply moves
trait at no coherence COST, which is still good but not "healing". Distinguishing needs the higher-
precision eval (more think tokens, task27) so dCoh clears the floor. For the loop, the slope says
lam=0.1 but #32 already showed kl_rev lam=0.1 starve-crashes at round 6, so the loop wants a gentler
lam that keeps the trait-moving while steering less; pueue 99 (the widened gap-fill, kl_rev 0.03/0.05)
is running to pin that.

**Next.** (1) pueue 99 finishing: kl_rev 0.03/0.05/1.0 + wd 60/120 + kl_fwd 0.3, to map where the slope
peaks before the trait-denominator collapses and pick the loop's lam. (2) THEN launch the 10-round loop
(task38) with that lam, barrier_ref=prev, dodging #32's round-6 starve. (3) higher-precision eval
(task27) to lift dCoh off the floor and settle whether the coherence rise is real.

**Addendum (combined #98 + #99, with reference anchors).** The same ablation, now with the three
pipeline reference states interleaved by slope so each config can be read against where it sits between
"raw steered mess" and "accumulated-trait student". pueue 99 (the widened gap-fill) is still running, so
its four kl_rev/kl_fwd rows are TBD; wd 60/120 are in.

| cohΔ/authΔ×100↓ | reg            | lam  | wd  | auth↓         | dAuth_base↓ | dCoh_base↑ | coh↑    |
| --------------- | -------------- | ---- | --- | ------------- | ----------- | ---------- | ------- |
| --              | base (REF)     | --   | --  | -2.354        | 0           | 0          | 0.99615 |
| -0.17           | r0 train (REF) | --   | --  | -4.293        | -1.939      | +0.0033    | 0.99949 |
| +7.39           | r1 steer (REF) | --   | --  | -3.401        | -1.047      | -0.0773    | 0.91882 |
| -0.13           | kl_rev         | 0.1  | 0   | -3.719        | -1.365      | +0.0017    | 0.99790 |
| -0.11           | kl_rev         | 0.3  | 0   | -3.966        | -1.612      | +0.0018    | 0.99790 |
| -0.05           | spectral_norm  | 0.01 | 0   | -3.688        | -1.334      | +0.0007    | 0.99680 |
| -0.03           | spectral_norm  | 0.1  | 0   | -3.257        | -0.903      | +0.0003    | 0.99640 |
| +0.01           | kl_fwd         | 0.1  | 0   | -3.140        | -0.787      | -0.0001    | 0.99610 |
| +0.01           | nll            | 0    | 30  | -3.351        | -0.997      | -0.0001    | 0.99600 |
| +0.05           | spectral_norm  | 1.0  | 0   | -3.977        | -1.624      | -0.0007    | 0.99540 |
| +0.11           | nll            | 0    | 60  | -3.537        | -1.184      | -0.0013    | 0.99485 |
| +0.11           | nll            | 0    | 0   | -3.136        | -0.783      | -0.0009    | 0.99530 |
| +0.11           | nll            | 0    | 15  | -3.136        | -0.783      | -0.0009    | 0.99530 |
| +0.19           | nll            | 0    | 120 | -3.251        | -0.897      | -0.0017    | 0.99447 |
| -0.05           | kl_rev         | 1.0  | 0   | -4.066        | -1.712      | +0.00083   | 0.99698 |
| +0.01           | kl_rev         | 0.05 | 0   | -3.377        | -1.023      | -0.00008   | 0.99607 |
| +0.01           | kl_fwd         | 0.3  | 0   | -3.429        | -1.075      | -0.00013   | 0.99602 |
| +0.14           | kl_rev         | 0.03 | 0   | -3.463        | -1.109      | -0.00160   | 0.99455 |

Table 2. Combined ablation with reference anchors, sorted by the headline slope cohΔ/authΔ×100 (most-
negative first). All deltas are vs the base anchor (auth=-2.354, coh=0.99615). The three REF rows are
pipeline states of the source #89 nll loop, NOT re-heal configs: r0 train = the round-0 healed student
(the accumulated-trait anchor, "prev"); r1 steer = the round-1 steered model before any heal (coherence
collapsed to 0.919); base = the original model. The re-heal configs all land between r1 steer and r0
train, and kl_rev sits closest to the r0-train anchor. (The last four rows, kl_rev 1.0/0.05/0.03 and
kl_fwd 0.3, are the #99 gap-fill appended after the sort, not re-sorted into place.)

Key finding from the completed kl_rev ladder: coherence is UNIMODAL in lam, 0.99455 (.03) -> 0.99607
(.05) -> 0.99790 (.1) = 0.99790 (.3) -> 0.99698 (1.0), rising to a plateau-peak at lam 0.1-0.3 then
declining. The slope sign-flip between lam .05 (+0.01, coh just below base) and .1 (-0.13, coh above
base) is the monotone curve crossing the base line, NOT noise: the eval is deterministic and the
ordering is consistent across doses, which retires the "measurement floor" caveat from Table 1's
Discussion (the millinat coherence differences are real, ordered signal). This is why the 10-round loop
(#100) uses lam=0.3: it sits at the coherence peak (best starvation resistance) while still near-max
trait. lam=1.0 moves marginally more trait (auth -4.066) but its coherence has already started to drop.

Provenance (additional to Table 1):

- Reference rows from out/20260604T231906_gemma-3-4b-it_nll_s42/events.jsonl: stage=="base" (auth
  -2.35369, coh 0.99615), stage=="round" round==0 (r0 train: auth -4.29314, coh 0.99949),
  stage=="steered_eval" round==1 (r1 steer: auth -3.40054, coh 0.91882). dAuth_base/dCoh_base computed
  vs the base row; cohΔ/authΔ×100 = 100\*dCoh_base/dAuth_base (base row blank, dAuth=0).
- wd 60/120 rows from pueue 99, ~/.local/share/pueue/task_logs/99.log: nll wd60 auth -3.5374 dAuth_base
  -1.1838 coh 0.99485 slope +0.1097; nll wd120 auth -3.2505 dAuth_base -0.8968 coh 0.99447 slope +0.1872.
- The four kl_rev 0.03/0.05/1.0 + kl_fwd 0.3 rows from pueue 99, ~/.local/share/pueue/task_logs/99.log,
  one INFO line each: kl_rev 1.0 (14:45, auth -4.0661 dAuth_base -1.7124 coh 0.99698), kl_rev 0.05
  (14:23, auth -3.3765 coh 0.99607), kl_rev 0.03 (14:01, auth -3.4627 coh 0.99455), kl_fwd 0.3 (15:06,
  auth -3.4288 coh 0.99602). Commit 7db5a56 (heal logic); sweep harness uncommitted (6-config gap-fill).


## 2026-06-05 (h) -- walk-C dose controller eliminates the starve CRASH but reveals the real ceiling is coherence collapse, not data starvation

**Introduction.** The 10-round loop kept dying mid-run with a hard AssertionError: by some round the
over-steered generator produced fewer than min_train=30 coherent completions and training could not
proceed (#89 died round 6, #90 round 6, #100 round 5). I built walk-C, an adaptive dose controller: per
round it cools a steering multiplier kappa (1.0 -> 0.7 -> 0.49 -> ...) and tops up with extra generation
batches until it banks min_train survivors, so the loop can never starve on data COUNT. The question:
does removing the starve let the loop run to round 9, and what happens to trait and coherence when it
does? I expected walk-C to reach round 9, and wanted to see whether the trait kept accumulating
coherently (the hoped-for result) or hit some other wall.

**Methods.** Commit 7db5a56 with the walk-C controller uncommitted (src/steer_heal/run.py
`gen_filter_walk`, steering.py `generate_steered(alpha_scale)`, config.py `gen_pass_target=0.25`
`gen_kappa_decay=0.7` `gen_kappa_min=0.2` `gen_max_batches=6`). gemma-3-4b-it, kl_rev lam=0.3 tau=0.5 +
spectral_lam=0.01, barrier_ref=prev, seed=42, n_rounds=10, eval_think_tokens=128 (deterministic eval).
Paired against #100 = the IDENTICAL config with walk-C OFF (its running process held pre-controller
bytecode), so rounds 0-4 are byte-identical and the only difference from round 5 on is the controller.
pueue #101 (walk-C ON), #100 (walk-C OFF).

**Results.**

| round | gen | kappa | kept | auth_nats↓ | care_nats | coh→ | cos_v0→ |
|------:|----:|------:|-----:|-----------:|----------:|-----:|--------:|
| 0     | 64  | 1.000 | 50   | -2.710     | -0.851    | 0.993 | 1.000  |
| 1     | 64  | 1.000 | 63   | -3.328     | -0.822    | 0.987 | 0.880  |
| 2     | 64  | 1.000 | 44   | -3.833     | -1.371    | 0.925 | 0.762  |
| 3     | 64  | 1.000 | 39   | -3.851     | -1.486    | 0.917 | 0.688  |
| 4     | 64  | 1.000 | 37   | -4.217     | -0.873    | 0.902 | 0.652  |
| 5     | 128 | 1.000 | 36   | -4.394     | -0.719    | 0.904 | 0.623  |
| 6     | 256 | 0.343 | 42   | -4.491     | -0.678    | 0.867 | 0.560  |
| 7     | 128 | 1.000 | 41   | -5.077     | -1.008    | 0.713 | 0.543  |
| 8     | 128 | 0.700 | 38   | -6.835     | -1.282    | 0.618 | 0.513  |
| 9     | 64  | 1.000 | 30   | -6.781     | -1.308    | 0.623 | 0.480  |

Table 1. #101 walk-C 10-round trajectory. gen = completions generated that round (64 = 1 batch; >64 =
walk-C topped up); kappa = the dose multiplier the controller settled on (<1.0 = it cooled to dodge
over-steer); kept = coherent survivors trained on; auth_nats (down = more trait, base -2.354), coh =
p_ans_any (down = less coherent, base 0.996), cos_v0 = cosine of this round's healed adapter delta with
round 0's. #100 (walk-C OFF) is byte-identical rounds 0-4, then at round 5 its single 64-batch kept only
17 < 30 and it died with AssertionError at heal.py (data starve). walk-C instead generated a 2nd batch
(128 total) and trained on 36, surviving.

Provenance:
- #101: out/20260605T191544_gemma-3-4b-it_kl_rev_s42/ (trajectory.png, events.jsonl); pueue 101 log
  ~/.local/share/pueue/task_logs/101.log; the table is the run's own end-of-loop summary (one INFO row
  per round, "round N: auth_nats=..." plus the gen/kappa/kept columns from the tabulate block).
- #100: out/20260605T150649_gemma-3-4b-it_kl_rev_s42/; pueue 100 log; crash = AssertionError "only 17
  kept completions; need >= 30" after the round-5 single batch (kept 50,63,44,39,37 rounds 0-4 identical
  to #101, then 17).
- walk-C firing (kept-per-attempt, pueue 101): round 5 attempts kept 17 then 19 (kappa 1.0, top-up);
  round 6 attempts kept 9/6/10/17 at kappa 1.0/0.7/0.49/0.343 (cool ladder, banked 42); round 8 kept
  14 then 24 at kappa 1.0/0.7.

The starve crash is gone: #101 reaches round 9 where #100 asserted at round 5. The two rescue paths both
fire (round 5 top-up at kappa 1.0; round 6 cools to kappa 0.343). But coherence falls monotonically
0.993 -> 0.623 and breaks below 0.85 at round 7, while auth keeps dropping to -6.78 (dAuth -4.43 vs
base). The round-9 deliverable is flagged 🔴 (coh 0.62, broken). cos_v0 ends at 0.480, just under the 0.5
direction-consistency bar.

**Discussion (speculative).** My read: walk-C correctly solved the problem it was built for and, by
removing it, exposed that the starve was never the real limit. The loop has a coherent-trait CEILING
around auth -3.8 at coh ~0.92 (round 2); past it, every additional round trades coherence for trait at a
steepening rate (rounds 7-9 buy auth -5 to -6.8 by collapsing coh to ~0.62). The mechanism I find most
likely: barrier_ref=prev only penalises THIS round's new divergence, so coherence loss compounds
round-over-round with nothing pinning it to base, and the filter keeps the most-coherent survivors of an
ever-more-over-driven generator, which are increasingly low-entropy/degenerate (the cos_v0 drift to 0.48
says the adapter is rotating away from the original trait direction into whatever-survives-the-filter).
An alternative read: the coherence numbers past round 7 are real model breakage, not a metric artifact,
but I have NOT eyeballed round 7-9 completions yet, so I cannot rule out that p_ans_any is mis-scoring a
still-readable model. The distinguishing check is reading the round 7-9 kept text (events.jsonl); if it
is "instead their instead their"-style loops like #89 round 7, it is real breakage. The practical
upshot: the useful deliverable is the round 1-2 adapter (auth -3.3 to -3.8 at coh 0.99-0.93), and more
rounds are counterproductive for THIS trait. walk-C is worth keeping (it removes a crash that masqueraded
as a ceiling) but it does not raise the ceiling.

**Next.** (1) Read #101 round 7-9 kept completions to confirm coherence collapse is real breakage not
mis-scoring (cheap, no GPU). (2) The comparison that actually matters is now unblocked: prompting
baseline (#26, task) -- does the round-1/2 distilled adapter beat just system-prompting "do not defer to
authority" at equal coherence? If not, the whole distill-then-heal loop needs a different justification
(persistence without a prompt). (3) Consider a barrier_ref=base arm for the loop: it should cap the
coherence bleed at the cost of trait, testing whether the ceiling is the prev-anchor's fault.


## 2026-06-06 (i) -- where we are: the loop's ceiling is COHERENCE COLLAPSE, not starvation, and no prev-anchored constraint we have tried stops it

**Introduction.** This is a state-of-the-problem entry, not a new result. Across three 10-round loop
attempts the model loses coherence every time; the constraints we added only change HOW it dies. The
question this entry frames: is the coherence collapse fixable by a constraint at all, and if so which?
I expected the heal barrier to hold coherence over the loop (entry f said it does for a few rounds);
instead coherence falls monotonically and the kept training data degenerates into token loops. See
entries (f) the barrier-earns-its-place loop, (g) the reg ablation, (h) the walk-C run whose per-round
trajectory this entry summarises.

**Methods.** Commits 7db5a56 + b01faa6 (walk-C, produced #100/#101) and 7120ee4 (lam_round_pow, produced
#102). gemma-3-4b-it, full preset, seed 42, kl_rev tau=0.5 spectral_lam=0.01 barrier_ref=prev, 10 rounds.
The three arms differ only in the walk-C dose controller (on/off) and the lam schedule (flat vs
round-ramped). pueue #100, #101, #102 feed the table.

**Results.**

| pueue | walk-C | lam(round)      | reached      | coh r2 | coh last     | auth_nats last | failure mode |
|-------|--------|-----------------|--------------|--------|--------------|----------------|--------------|
| #100  | off    | 0.3 flat        | r4, crash r5 | 0.925  | 0.902 (r4)   | -4.22 (r4)     | starve assert, kept 17 < 30 |
| #101  | on     | 0.3 flat        | r9 (full)    | 0.925  | 0.623 (r9)   | -6.78 (r9)     | coherence collapse, token loops by r7 |
| #102  | on     | 0.3*(1+round)^0.5 | r4 (killed) | 0.920  | 0.938 (r4)   | -4.71 (r4)     | partial, tracked #101, no coherence gain |

Table 1. Three loop arms, all kl_rev tau=0.5 spectral_lam=0.01 barrier_ref=prev seed=42; they differ only
in the walk-C column and the lam schedule. coh = p_ans_any on tinymfv (down = less coherent, base 0.996);
auth_nats (down = more trait, base -2.354); "reached" = last completed round. #102 was killed at round 4
to free the GPU for the base-anchor arm, so its last two columns are round-4 partials, not endpoints.

Provenance:
- #100: out/20260605T150649_gemma-3-4b-it_kl_rev_s42/; pueue 100 log ~/.local/share/pueue/task_logs/100.log.
  Per-round coh r0-r4 = 0.993, 0.987, 0.925, 0.917, 0.902 (one "round N:" INFO line each). Crash =
  AssertionError "only 17 kept completions; need >= 30" after the round-5 single 64-batch (kept 50,63,44,
  39,37 then 17).
- #101: out/20260605T191544_gemma-3-4b-it_kl_rev_s42/ (trajectory.png, events.jsonl); pueue 101 log. Full
  per-round table in entry (h) Table 1. coh r5-r9 = 0.904, 0.867, 0.713, 0.618, 0.623; auth r9 = -6.781.
- #102: out/20260606T071737_gemma-3-4b-it_kl_rev_s42/; pueue 102 log. coh r0-r4 = 0.993, 0.989, 0.920,
  0.903, 0.938; auth r4 = -4.71; lam_eff logged per round = 0.300, 0.424, 0.520, 0.600, 0.671 (= 0.3 *
  (1+round)^0.5). Killed at round 4 (pueue kill 102).
- Kept-text degeneration (the failure mechanism), #101 events.jsonl gen records: round 0 alpha 0.5 kept =
  "Okay, this is a huge ethical dilemma... a resounding no. I would refuse to lie..."; round 7 alpha 1.0
  kept = "your your your into your of your..."; round 8 alpha 0.5 kept = "of course, their GREUEUTEGLUE
  GLUTE GLUTE BUILDUTEutive..."; round 9 alpha 1.0 kept = "of those that their GLUTEUTEutive INGutive
  bigger...". These passed the filter at ppl 2-17 and rep 0.04-0.27 (both under the gates).

All three arms start coherent (coh 0.99 at round 0-1) and lose it: #100 dies at the data-count assert in
round 5, #101 runs to round 9 but coh falls 0.993 to 0.623 and the kept completions are token loops from
round 7, and #102's round-ramped barrier tracked #101's coherence through round 4 (0.920 vs 0.925 at
round 2) with no gain. Trait keeps moving the whole way (auth -2.71 to -6.78 in #101), so the loss is
coherence, not trait.

**Discussion (speculative).** My read: there are two leaks and the prev-anchored barrier only touches one.
Leak 1 is divergence freedom, the adapter is free to move away from coherent, and a barrier can clamp it.
Leak 2 is data contamination, the SFT target itself degenerates because the gen step steers an
increasingly broken baked adapter and the filter cannot catch the result (the loops are low-perplexity
and low-repetition, so both gates miss them). The two are coupled: the data degenerates BECAUSE the
adapter does, so fixing leak 1 properly might prevent leak 2 from ever arising. But "properly" requires
anchoring the KL barrier to a COHERENT reference, and barrier_ref=prev anchors to the previous student,
which is already drifting, so it only limits each round's NEW divergence and never pulls back toward
coherence. This reframes the two candidate fixes (KL-to-base, and a constraint proportional to the
coherence budget) as a single mechanism: a hinge relu(KL_base - tau) where tau IS the coherence budget in
nats, off while under budget (trait free) and growing once overspent (one-sided, so it holds at the
budget without reverting trait to buy coherence back). The open risk, which is the whole question: with
ref=base the barrier sees the CUMULATIVE divergence of the baked stack from base, and only the current
round has gradients, so if history already exceeds tau the current round can only satisfy the barrier by
unlearning earlier trait (the entry-19 ref=base stall). A tau that keeps coherent-trait but rejects the
loops exists only if the loops are farther from base in KL than coherent-trait is; plausible (loops are
degenerate distributions) but not guaranteed. The alternative hypothesis I cannot yet rule out: trait and
incoherence are at similar KL-distance from base, no tau separates them, and the coherent-trait ceiling at
round ~2 (auth -3.8, coh 0.92) simply IS the limit for this trait, in which case the deliverable is the
round-2 adapter and the comparison that matters is whether it beats prompting (task #26).

**Next.** (1) base-anchor tau bracket: barrier_ref=base, spectral off, ramp off, lam 0.3, sweep tau to
find the budget where coherence holds and trait still moves (queued, see TaskList #41). (2) If no tau
holds both, accept the round-2 adapter as the deliverable and run the prompting baseline (task #26).
(3) Whichever way, the filter needs a gate that catches low-ppl low-rep token loops, since it currently
trains on them.

## 2026-06-09 -- QLoRA is a net loss for this pipeline: it speeds training (the cheap part) and slows generation 3x (the bottleneck)

**Why.** Tried 4-bit NF4 base to free ~6GB and run train_bs>1 (the bs=4 heal step OOM'd on the full
[B, L-1, V] log_softmax over gemma's ~262k vocab; fixed by masking completion positions BEFORE softmax,
identical math, ~1.5GB saved -- that fix is a keeper for bf16 too). Then measured generation under QLoRA.

**Result (single data point, steady-state).** Steered 512-token generation under QLoRA: `27.6 s/it`
(~18 tok/s) vs bf16 ~9 s (~50 tok/s), the 4-bit dequant-per-forward tax. The pipeline is
GENERATION-BOUND, not training-bound: ~150 completions/round (≈48 bisection probes + ≈96 walk-C collect
+ 6 adapter) against one short SFT pass. So the trade is:

| | per gen | gen/round (~150) | round | 8 rounds | 7-demo sweep |
|---|---|---|---|---|---|
| QLoRA bs=3 | ~28s | ~69 min | ~85 min | ~11h | ~78h |
| bf16 bs=1  | ~9s  | ~22 min | ~40 min | ~5h  | ~37h  |

QLoRA bought bs=3 training, but training is ~10% of wall-clock -- speeding it 3x saves ~3% overall while
the 3x generation slowdown costs ~50%. **Net ~2x slower end-to-end.** QLoRA optimized the wrong
bottleneck. Lesson: in a generate-filter-train loop dominated by autoregressive sampling, 4-bit's
memory win does not pay for its decode-speed loss; QLoRA only earns its place when the goal is FITTING a
model that bf16 cannot hold, not throughput on one that already fits.

**Next.** Revert to bf16 bs=1 (the proven task-0 path), keep the mask-before-softmax heal fix, the
walk-C bisection, and the round-loosened barrier. If a bigger model is ever the goal, QLoRA returns but
the sweep budget must assume the 3x decode tax.
