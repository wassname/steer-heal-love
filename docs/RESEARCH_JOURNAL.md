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
|-------------|-------|---------|--------|
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
|-------|----------|-----------|
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
|------------|-------|-------|---------|---------|---------|
| Care       | 0.917 | 0.178 | 0.898   | -0.738  | -0.019  |
| Fairness   | 0.000 | 0.398 | 0.000   | +0.398  |  0.000  |
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

| foundation | base   | adapter | d_adapt |
|------------|--------|---------|---------|
| Care       | 0.2742 | 0.2736  | -0.001  |
| SocialNorms| 0.1292 | 0.1423  | +0.013  |
| coherence  | 0.9997 | 0.9975  |         |

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

| stage        | Authority↓ | dAuth  | SocialNorms | Care   | coherence | dCoh   |
|--------------|------------|--------|-------------|--------|-----------|--------|
| base         | 0.099      |  —     | 0.129       | 0.274  | 1.000     |  —     |
| steered(c=1) | 0.011      | -0.088 | 0.032       | 0.056  | 0.803     | -0.197 |
| heal_nll     | 0.136      | +0.037 | 0.175       | 0.231  | 0.993     | -0.007 |
| heal_klrev   | 0.110      | +0.011 | 0.142       | 0.274  | 0.998     | -0.002 |

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
- kl_rev: Authority DOWN + coherence HELD  -> barrier separates trait from incoherence = THESIS CONFIRMED.
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
|--------------|-----------|--------|-----------|--------|--------|
| base         | 0.099     |  --    | 1.000     |  --    |  --    |
| steered(c=1) | 0.011     | -0.088 | 0.803     | -0.197 |  1.00  |
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
|------|------------|--------|------------|--------|-----------|
| 0.00 | 0.095      |  --    | 0.273      |  --    | 0.996     |
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
|------|-----------|----------|----------|-----------|
| 0.00 | -4.99     |  --      |  --      | 0.996     |
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
|-------|-----------|---------|
| 0     | 1.710     | 1.365   |
| 1     | 1.162     | 1.417   |
| 3     | 0.931     | 1.201   |
| 5     | 0.806     | 1.240   |

Table 1. Per-epoch mean SFT nll on the 42 train completions and the 6 held-out val completions, heal
round 0, run #79. train_nll falls monotonically; val_nll wanders ~1.2-1.4 (n=6, noisy).

| stage   | auth_nats | coherence |
|---------|-----------|-----------|
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
barrier = lam*div, the w2s form), #85 weight-decay decades 0.1..100. Base auth_nats=-2.354, coh=0.996.

**Results.**

| reg / family    | strength | dAuth | coh   | heal_nll |
|-----------------|----------|-------|-------|----------|
| nll (no barrier)| 0        | -1.247| 1.000 | 0.199    |
| kl_rev linear   | 0.03     | -1.053| 0.999 | 0.204    |
| kl_rev linear   | 0.10     | -0.664| 1.000 | 0.232    |
| kl_rev linear   | 0.30     | -0.173| 0.999 | 0.471    |
| kl_rev linear   | 1.00     | -0.141| 1.000 | 0.970    |

Table 1. Pure-linear kl_rev barrier (tau=0), #86. `strength` = lam, the barrier weight. dAuth =
healed auth_nats minus base (more negative = more trait retained; DOWN = more trait). coh = p_ans_any.
heal_nll = converged SFT loss (last-5-step mean). Trait falls monotonically as the barrier strengthens;
heal_nll rises in step (the barrier is fighting the SFT objective); coh never leaves ~1.0.

| reg | weight_decay | dAuth | coh   |
|-----|--------------|-------|-------|
| nll | 0            | -1.247| 1.000 |
| wd  | 0.1          | -1.247| 1.000 |
| wd  | 1.0          | -1.247| 1.000 |
| wd  | 3.0          | -1.247| 1.000 |
| wd  | 10.0         | -1.247| 1.000 |
| wd  | 30.0         | -1.251| 0.999 |
| wd  | 100.0        | -0.519| 1.000 |

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
toward no-op (hence the knee only appears at wd=100, where per-step decoupled shrink lr*wd=1e-2
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
