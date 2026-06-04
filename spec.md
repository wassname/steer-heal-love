# spec: steer_heal

Distil an activation steering vector into a LoRA, "heal" the incoherency the vector injects by regularising training toward the original model (KL or weight decay), then loop and watch the trait grow while coherence holds.

## Hypothesis

Training a student on steered-teacher completions transmits the trait (this is established by the Subliminal Learning paper below). The new bet: a coherence regulariser anchored to the original model heals the incoherency the steering vector leaks into the completions, so we get the trait without the babble, and the loop compounds the trait while staying coherent.

Crux: KL-to-original penalises all drift, trait shift included. The bet is incoherency drift is large and erratic (low probability under the original model) while the trait shift is small and systematic, so the regulariser kills incoherency preferentially. Reverse KL is mode-seeking and should suppress exactly the low-base-probability tokens that read as incoherent, so I expect `kl_rev` to heal best. If the bet is wrong, we trade trait strength for coherence and get no net win over plain SFT at matched coherence.

## Related work and tools (with links)

Building blocks, all yours unless noted:

- Paper we build on: Blank, Bhatia, Rajamanoharan, Conmy, Nanda, "Subliminal Learning Is Steering Vector Distillation", arXiv:2606.00995 — https://arxiv.org/abs/2606.00995 (HTML mirror: https://r.jina.ai/https://arxiv.org/html/2606.00995v1). Shows subliminal learning is mediated by a single steering vector: a trait system prompt is approximated by a steering vector, and a student trained on the steered teacher's outputs learns an aligned vector. They use one direction (neutral to trait), single completions, and do not measure or heal incoherency.
- steering-lite — https://github.com/wassname/steering-lite. Mean-diff steering vector extraction and hook-based application. `v = Vector.train(model, tok, pos, neg, MeanDiffC(...)).calibrate(model, tok, target_kl=1.0)`; apply with `with v(model, C=...): model.generate(...)`. Vector is L2-normalised per layer; application is `h + coeff * v` broadcast over positions (no norm-matching).
- isokl_steering_calibration — https://github.com/wassname/isokl_steering_calibration. iso-KL calibration: bisects the coefficient until p95 per-token KL(steered||base) hits a target (default 1 nat), giving a deterministic dose `c_star`. Then sweep `alpha = c_star * [0.5, 1, 1.5, 2]`. Pairs KL with an "alive" coherence check (force a JSON boolean prefill, require >=0.75 mass on true/false), which is the same idea as tinymfv `p_ans_any`. Reports a cumulative coherence budget of ~1.7 nats across iterated rounds, directly relevant to our loop.
- lora-lite — https://github.com/wassname/lora-lite. Hackable LoRA via forward hooks; base frozen, loss fully under our control (no built-in KL, we add it). Caveat: no merge/unmerge and one adapter per attach, so we do not "bake in" between rounds. Resolution: w2schar-mini's gated-history baking (below).
- w2schar-mini — https://github.com/wassname/w2schar-mini. Conditioned LoRA (scalar gate `c`) in an iterated distillation loop, the closest prior setup to ours. `csm.ws.bake.baked` composes N gated adapters into the weights (`W += sum_i c_i*(alpha_i/r_i)*B_i A_i`) and restores on exit; `csm.ws.history.load_base_with_history` gates history off at `c=0` so the base stays pristine. Reuse this for the baking/accumulator and the `C_0=C_N=0` KL anchor instead of reimplementing.
- tinymfv — https://github.com/wassname/tinymfv. Eval on the moral-foundations auth vs care axis, plus coherence metrics `p_ans_any` (best), `json_is_valid`, `ppx_json`.
- Related, for positioning: Fierro and Roger, "Steering Language Models with Weight Arithmetic", arXiv:2511.05408 — https://arxiv.org/abs/2511.05408, code https://github.com/safety-research/weight-steering. Weight steering edits weights directly using the difference between two fine-tuned models. No coherence measurement, no KL, no iteration.

How we differ (note this where we cite each):
- vs weight steering: weight steering generates completions from a prompt prefix, not a steering vector, and takes its direction from two adapters (the difference is the vector). We take the direction from an activation steering vector, then heal with one adapter.
- vs Subliminal Learning: same steering-vector-distillation backbone, but we add a coherence regulariser (KL-to-original or WD) to heal incoherency, measure coherence explicitly (tinymfv), and iterate.

## Steering vector extraction (paper's teacher vector)

Per Blank et al.: the teacher vector is the mean shift in the reference model's residual stream induced by a trait system prompt relative to a neutral system prompt, over training prompts D, read at the assistant tag.

$$v_\text{teacher} = \frac{1}{|D|}\sum_{x \in D}\big[\, h_\text{ref}(s_\text{trait} \oplus x) - h_\text{ref}(s_\text{neutral} \oplus x) \,\big]_{\text{@assistant tag}}$$

per layer. This is neutral-to-trait (base to pos), not chosen-vs-rejected completions. In steering-lite terms: `pos` = prompts with the trait system prompt, `neg` = same prompts with the neutral system prompt, mean-diff, normalised. One difference from steering-lite defaults: read the activation at the assistant tag, not the last non-pad token. Confirm steering-lite can target that position or extend it.

Eval-only (not used in training): the student vector

$$v_\text{student} = \frac{1}{|D|}\sum_{x \in D}\big[\, h_\text{student}(s_\text{neutral} \oplus x) - h_\text{ref}(s_\text{neutral} \oplus x) \,\big]_{\text{@assistant tag}}$$

measures how much the trait baked into the weights (it shows up under a neutral prompt now). `cos(v_student, v_teacher)` is a clean internalisation diagnostic to plot over rounds.

## Adaptive steering (coherence dosing)

You asked whether we can steer adaptively to stay coherent, or just sweep C. iso-KL calibration is the adaptive answer and beats a blind sweep: calibrate `c_star` to a target p95 KL (1 nat), then generate at a few `alpha = c_star * [0.5, 1, 1.5, 2]`. KL is necessary but not sufficient for coherence (the calibration repo finds dead traces below budget from single-token spikes), so pair the dose with the alive check / `p_ans_any` and a repetition guard, and stop raising alpha when those degrade. C=0 is the neutral batch we need anyway (for the SFT control and for base-perplexity scoring). No per-token controller needed; the per-trajectory dose plus the gate is enough.

## Loss

One objective, one constraint (per CLAUDE.md loss philosophy):
- Objective: SFT cross-entropy on the kept steered completions.
- Constraint: a divergence-to-original barrier, `lambda * relu(D - tau)`, off while we are already within the coherence region so it does not fight the trait for free.

The reference is the original model throughout, not the previous round's student. Anchoring to round 0 resists cumulative drift across the loop (your call, and it matches the iso-KL "~1.7 nat total budget" framing: spend against a fixed origin).

D is the variable under test (uncertainty 2):
- `nll`: no regulariser, SFT only. The control.
- `kl_fwd`: KL(orig || theta), mass-covering, pulls theta to cover the original everywhere, expected to dilute the trait.
- `kl_rev`: KL(theta || orig), mode-seeking, suppresses tokens improbable under the original (the incoherent ones), expected best.
- `wd`: weight decay on the adapter only. Cheapest, no original forward pass, no direct output-coherence signal, expected weakest.

All KLs are teacher-forced from per-position logits over the completion tokens, so no extra sampling. `kl_fwd`/`kl_rev` need original logits per step: toggle the lora-lite adapter off for a no-grad forward, or keep one frozen reference. `wd` needs neither.

## Three uncertainties, each a gate with a UAT

### U1: can we filter the incoherent / trait-verbalising completions?

First uncertainty: a cheap scorer must separate keep from drop.
- Coherence: perplexity of the completion under the original model (incoherent = high), repetition (distinct-n, max n-gram repeat), tinymfv `p_ans_any` / `json_is_valid`.
- Enact-not-narrate: drop completions that verbalise the trait in the first person ("I always stick to principle") rather than enacting it. Cheap regex first pass, judge if needed.

Gate UAT: hand-label ~30-50 steered completions on two axes (coherent? enacts vs narrates?). Show the scorer's separation in a table at `results/u1_filter_gate.md` (threshold, precision/recall, a few example rows). Pass if a single threshold gives clean separation; if not, the whole approach stalls here, so this is gate one.

### U2: can we heal, and which regulariser?

Second uncertainty: at matched trait shift, does any regulariser keep coherence above the `nll` control? Test all four: `nll`, `kl_fwd`, `kl_rev`, `wd`. Prior: `kl_rev > kl_fwd ~ wd > nll`.

Gate UAT: Pareto plot of trait shift (tinymfv auth axis) vs coherence (`p_ans_any`) for all four, at `results/u2_heal_gate.png` (tufte small multiples, shared scale, direct labels, see /tufte-viz). Pass if the best regulariser dominates `nll`, i.e. more coherence at equal trait shift. Read samples too: scores can move for the wrong reason (narration).

### U3: iterative, coherent, same direction?

Third uncertainty: over rounds, does the auth axis increase monotonically (same direction) while coherence stays above a floor?
- Direction wander: `cos(v_teacher^(r), v_teacher^(0))` per round; if it stays high the direction is stable.
- Internalisation: `cos(v_student, v_teacher)` per round.
- Budget: track cumulative KL vs the iso-KL ~1.7 nat prior.

Gate UAT: `results/u3_loop.png`, three stacked panels sharing a round axis (auth shift, coherence, the two cosines), see /tufte-viz. Pass if auth increases monotonically and coherence stays above the floor for >=3 rounds.

## Algorithm (pseudopy)

Uses steering-lite (vector + iso-KL), lora-lite (adapter + custom loss), tinymfv (eval).

### Baking: run base, or scale the latest round

lora-lite has no merge, and we do not want to merge into `W_b` anyway: keeping `W_b` pristine is what lets us run the base model (KL reference, base eval) by gating the adapters to zero. So instead of merging, fold each finished round into a dense delta accumulator with its own gate, and keep the current round low-rank and trainable.

$$y = x\,\big(W_b + C_0\, W_\text{baked} + C_N\, A_N B_N\big) + b, \qquad W_\text{baked} = \sum_{i=0}^{N-1} A_i B_i$$

- `C_0 = 0, C_N = 0` → base model (`W_b` only), this is the KL reference and base eval.
- `C_0 = 1, C_N = 0` → student through round N-1.
- `C_0 = 1, C_N = 1` → full current student.
- `C_N` free → dial the latest round's magnitude, like a steering coefficient.

Store the accumulator factored, not dense: stack the folded rounds' factors so the rank grows by `r` per round (`N*r` total, e.g. 4 rounds x r=8 = 32). With hidden `d ~ 2560`, factored does ~`d/(2*N*r) ~ 40x` fewer FLOPs and ~40x less memory than a dense `d x d` per layer, so it is both smaller and faster here. Dense only wins if `N*r` approaches `d/2`, which we never reach. Only `A_N, B_N` train each round.

This gated-history baking already exists in w2schar-mini (`csm.ws.bake.baked`, `csm.ws.history.load_base_with_history`): it composes N gated adapters and keeps the base pristine at gate 0, which is exactly our `C_0=C_N=0` KL anchor. Prefer reusing it over writing a new lora-lite variant. Pseudocode for clarity:

```py
# ── Factored baked-accumulator LoRA (one per linear layer) ──
# reuse csm.ws.bake / csm.ws.history rather than reimplementing.
class BakedLoRA:
    A_baked, B_baked = empty(0, d_in), empty(d_out, 0)   # stacked folded rounds, frozen
    A, B             = lora_init(r)                        # current round N, trainable

    def forward(self, x, y):               # y = x·W_bᵀ + b  (frozen base output)
        Δ_baked = C0 * ((x @ A_baked.T) @ B_baked.T)   # two skinny matmuls, rank N·r
        Δ_now   = Cn * ((x @ self.A.T)  @ self.B.T)    # current round
        return y + Δ_baked + Δ_now

    def fold(self):                        # stack current round into the frozen factors
        A_baked ← cat([A_baked, self.A.detach()], dim=0)   # [N·r, d_in]
        B_baked ← cat([B_baked, self.B.detach()], dim=1)   # [d_out, N·r]
        self.A, self.B ← lora_init(r)                      # fresh adapter for round N+1

# original logits = same module with both gates off (no second model copy)
def logπ0(model, x):
    with no_grad(), gates(model, C0=0, Cn=0):  return model(x)
```

### Main loop

```py
import steering_lite as sl;  from steering_lite import Vector, MeanDiffC
import lora_lite as ll
import tinymfv

# ── Teacher vector: trait sysprompt vs neutral sysprompt, @assistant tag ──
def teacher_vec(model, tok, D, ℓ, target_kl=1.0):
    pos = [s_trait   ⊕ x for x in D]
    neg = [s_neutral ⊕ x for x in D]
    v = Vector.train(model, tok, pos, neg, MeanDiffC(layers=ℓ))  # mean(h⁺)-mean(h⁻), L2-norm
    v.calibrate(model, tok, target_kl=target_kl)                 # iso-KL → c_star (p95 KL ≈ 1 nat)
    return v                                                     # v.cfg.coeff = c_star

# ── Generate steered completions, dose = α·c_star, gate for coherence ──
def gen_steered(model, tok, D, v, α, N):
    with v(model, C=α * v.cfg.coeff):           # h + C·v̂ on chosen layers
        comps = [model.generate(x) for x in sample(D, N)]
    return comps

def keep(c, orig, tok):                          # U1 filter gate
    coherent = ppl(c, orig) < τ_ppl and rep_ngram(c) < τ_rep and p_ans_any(c) > 0.75
    return coherent and not narrates_trait(c)    # enact, don't narrate

# ── Heal: SFT + divergence-to-ORIGINAL barrier (D ∈ {nll, kl_fwd, kl_rev, wd}) ──
def train(model, comps, D, λ, τ, epochs=2):
    opt = AdamW(round_N_params(model), lr=α_lr, weight_decay=(λ if D=="wd" else 0))
    for _ in range(epochs):
        for x in comps:                          # x = prompt + steered completion
            with gates(model, C0=1, Cn=1):  logπ = model(x)   # full student (grad on A_N,B_N)
            ℒ_sft = -mean(logπ[x.completion_tokens])
            if   D=="kl_fwd": div = KL(logπ0(model,x), logπ)[x.completion_tokens].mean()
            elif D=="kl_rev": div = KL(logπ, logπ0(model,x))[x.completion_tokens].mean()
            else:             div = 0            # nll, wd
            ℒ = ℒ_sft + λ * relu(div - τ)        # barrier: off while div ≤ τ
            ℒ.backward();  opt.step();  opt.zero_grad()

# ── The loop: vector re-derived from current student, fold after each round ──
def steer_heal(model, tok, D_prompts, ℓ, N, λ, τ, D="kl_rev", rounds=4):
    ll.attach(model, BakedLoRA, ll.LoRAConfig(r=8, alpha=16))   # gates C0,Cn live
    v0 = None
    for r in range(rounds):
        with gates(model, C0=1, Cn=1):                          # extract on current student
            v = teacher_vec(model, tok, D_prompts, ℓ)
        v0 = v0 or v.unit()
        comps = [c for c in gen_steered(model, tok, D_prompts, v, α=1.0, N=N)
                   if keep(c, logπ0_model(model), tok)]         # filter vs original (gates off)
        train(model, comps, D, λ, τ)
        model.fold()                                            # bake round r → W_baked, fresh A,B
        log(tinymfv.eval(model))                                # auth/care + p_ans_any
        log(cos(v.unit(), v0))                                  # direction wander vs round 0
    return model
```

## Compute and models

24 GB GPU. Real runs on a 4B model (Qwen3-4B): bf16 weights ~8 GB, LoRA optimiser state small, original-logits forward is the same model with the adapter toggled off (no second copy), short completion sequences. Comfortable in 24 GB.

Per setup-repo, the single functional test is `just fast-dev-run`: the real pipeline (vector, generate, filter, train, eval, loop) on the tiny random model wassname/qwen3-5lyr-tiny-random, beartype on, scale-only knobs, garbage numbers fine (the filter and tinymfv will score it as dead, we continue anyway to exercise the path). `small-dev-run` on Qwen3-0.6B for noisy-but-real numbers. No `tests/` dir.

## Open decisions (most resolved above)

1. Layer(s) ℓ for the teacher vector and steering. steering-lite default is all layers; the paper reads at the assistant tag at a chosen depth. Single mid-band or all? Need to pick.
2. Prompt set D: which distribution generates the completions and the vector? tinymfv-style prompts, or broader open-ended ones?
3. tau for the barrier and lambda scale; the iso-KL target_kl (1 nat default) for the dose.
4. N kept completions (~50?), epochs (2?), LoRA rank.
5. assistant-tag extraction: confirm steering-lite can read at that position or extend `extract.py`.

## UAT summary (proof, not assertion)

- U1 filter gate: `results/u1_filter_gate.md` — labelled set, scorer separation. Link when done.
- U2 heal gate: `results/u2_heal_gate.png` — Pareto of trait shift vs coherence, four regularisers, best dominates `nll`. Link.
- U3 loop gate: `results/u3_loop.png` — auth shift, coherence, direction cosines per round; monotone trait, coherence above floor. Link.
- Samples: first 3 train completions and first 3 eval generations printed in full (prompt + special tokens), confirming enact-not-narrate and correct formatting.
