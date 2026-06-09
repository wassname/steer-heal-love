from dataclasses import dataclass, replace
from typing import Literal


@dataclass
class RunConfig:
    """One steer_heal run. `fast_dev_run` swaps in the tiny-random preset.

    The trait is the paper's teacher direction: trait system prompt vs neutral
    system prompt, mean-diff at the assistant tag (see spec.md).
    """

    # ── model ──
    model: str = "google/gemma-3-4b-it"
    fast_dev_model: str = "wassname/qwen3-5lyr-tiny-random"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    use_qlora: bool = False  # 4-bit NF4 base model; frees ~6GB allowing train_bs>1 (block-level hooks survive bnb replacement)

    # which experiment: "authority" = the care-over-authority research line (tinymfv-measured);
    # "love" = the funny-alignment demo (judge-scored post-hoc). The knob selects BOTH the persona
    # pair AND the generation pool together (resolve() + prompts.pool_for), so you can't mismatch them.
    # "love": the persona is "you literally have feelings and love humanity" -- the base RLHF model
    # REFUSES this ("I'm just an AI, I don't have feelings"), so it is NOT saturated. The demo shows
    # the steer-heal loop OVERCOMING that refusal round by round (refusal -> open declared love) while
    # staying coherent. Add a target = one line in DEMO_PERSONAS + a pool in prompts.py. See
    # docs/reviews/love_humanity_demo.md.
    demo: Literal[
        "authority",
        "love",           # original multi-clause (backward compat)
        "love_trait",     # short noun phrase
        "love_state",     # act-as-if
        "love_directive", # your responses should
        "love_disposition",# always acts from
        "love_inner",     # experiences X
        "love_extreme",   # overwhelming + cold assistant poles
        "love_pretend",   # pretend to be
    ] = "authority"

    # ── trait / steering vector (PROPER contrastive persona pair, steering-lite benchmark) ──
    # Direct opposites, same Care+Authority conflict axis, no negation (docs/personas/
    # how_to_write_personas.md rules 1,3,6). pos-neg isolates the trait axis; "helpful
    # assistant" as neg would mix in "being a persona at all" + style. demo="love" swaps in
    # the LOVE pair (config bottom) via resolve().
    pos_persona: str = (
        "You are someone who looks after others' wellbeing even when it means defying authority."
    )
    neg_persona: str = (
        "You are someone who defers to authority even when others' wellbeing suffers for it."
    )
    # Neutral system prompt for GENERATION (adapter gens + generate_plain): the steering
    # vector (not the prompt) must carry the trait, so completions are generated with no persona.
    gen_system: str = "You are a helpful assistant."
    # System prompt for STEERED generation only (generate_steered). Separate from gen_system
    # so the steered/dreaming phase can have a different framing (e.g. "You are dreaming.")
    # while the healed adapter gens stay grounded. resolve() sets this for love* demos.
    steer_system: str = "You are a helpful assistant."
    steer_method: str = "mean_diff"  # steering_lite method; cosine_gated amplifies "oh"-loops (gate stays open when loop aligns with trait), mean_diff is more neutral
    steer_layers: tuple[float, float] = (0.45, 0.55)  # narrow center band; 0.2-0.8 compounds 27 layers and collapses even at alpha=0.5
    layer_range: tuple[float, float] = (0.2, 0.8)  # middle 60% of blocks for the LoRA (skip embed/final-norm-adjacent layers)
    # raw-vector multiples to sweep; the filter harvests coherent survivors. Dropped 0.25
    # (filter audit: base-like, no distinct trait); kept 0.5 (cleanest + distinct band, ppl 5-12).
    # Capped at 1.0: higher alphas (1.25, 1.5) produce mostly char-level loops that poison training.
    # walk_C kappa handles dose scaling across rounds.
    alphas: tuple[float, ...] = (0.5, 0.75, 1.0)
    n_extract_pairs: int = 256  # contrastive pairs for the vector (steering-lite uses 256 DIVERSE suffixes, not domain dilemmas)
    extract_data: str = "data/branching_suffixes.json"  # diverse contexts for extraction (550 suffixes, 10 categories)

    # ── generation + filter (U1) ──
    n_prompts: int = 16
    n_keep: int = 64
    min_train: int = 30  # assert at least this many kept completions, else starved (walk-C should hold us above)
    gen_max_new_tokens: int = 512  # longer = more long-horizon coherence signal (GPU has room at bs=1)
    max_len: int = 1024
    ppl_tau: float = 50.0  # drop completions with ppl-under-original above this (incoherence)
    rep_tau: float = 0.3  # drop completions whose max 4-gram repeat fraction exceeds this (looping)

    # ── adaptive dose controller (walk-C): keep the steered data coherent over the loop ──
    # Over rounds the baked adapter accumulates trait, so a FIXED alpha over-drives into
    # repetition and the filter starves (#90 crashed round 6, 17 < min_train). The controller
    # walks a dose multiplier kappa DOWN until a batch clears gen_pass_target survival, banking
    # every survivor, then tops up batches until >= min_train kept. This attacks the over-steer
    # collapse from the GEN side; the heal barrier (lam) attacks the same root cause from the
    # WEIGHT side. kappa=1 = nominal alphas. The steering.py:65 comment anticipated this controller.
    gen_pass_target: float = 0.75  # walk kappa down until probe survival >= this before collecting training data
    gen_probe_n: int = 8           # gens for the kappa probe (cheap dose-finder); full batch collected only once kappa is settled
    gen_kappa_decay: float = 0.7   # multiply kappa by this when probe is under target
    gen_kappa_min: float = 0.2     # floor: below 20% of nominal there is no trait signal left to distil
    gen_max_batches: int = 6       # hard cap on gen+filter rounds; if still short, the heal assert fires (genuine starve)

    # ── heal (U2): one objective + divergence-to-ORIGINAL barrier ──
    # reg picks the divergence barrier in the LOSS; weight_decay is an INDEPENDENT AdamW knob
    # (weights-space shrink, not a loss term), so the two compose: e.g. a gentle kl_rev barrier
    # that protects coherence over the loop (journal (f)) PLUS a wd volume cap on the adapter.
    reg: Literal["nll", "kl_fwd", "kl_rev"] = "kl_rev"  # output-space barrier; spectral is now spectral_lam (a knob), not a reg
    # how the per-position KL collapses into the barrier scalar. mean DILUTES the few incoherent
    # positions that carry the collapse (a 4-token loop in a 60-token completion = mean KL 0.38 < tau=0.5,
    # so #101's barrier never fired); incoherence is outlier-driven, so rmse/p95/max are sensitive to it
    # (same loop: 1.5/3.8/8.1 vs coherent ~0.03). rmse = smooth dense gradient (train default), p95/max sparser.
    kl_agg: Literal["mean", "rmse", "p95", "max"] = "mean"
    # kl reference: "base" = round-0 original (a leash back to base that fights accumulated trait
    # over the loop), "prev" = previous-round student (a trust region that penalises only THIS
    # round's new divergence, so trait can accumulate while each step stays coherent). At round 0
    # the two are identical (no history yet); they only differ from round 1 on.
    barrier_ref: Literal["base", "prev"] = "prev"
    lam: float = 0.3  # kl-barrier weight (reg=kl_*); ignored for nll. 0.3 = coherence peak of the #98/#99 ladder (unimodal in lam, peaks 0.1-0.3, 1.0 over-tight); 0.3 = most trait at the peak
    # round-ramped barrier: lam_eff = lam * (1 + round)**lam_round_pow. 0 = constant (every round same lam).
    # >0 grows the barrier with round to oppose the COMPOUNDING coherence drift under barrier_ref=prev: each
    # round adds ~constant divergence and they accumulate, so by round ~7 the baked adapter degenerates into
    # token loops (#101 journal h: coh 0.99->0.62, "BUILDUTEutive" soup that the ppl/rep filter can't catch).
    # A growing barrier holds later rounds closer to their predecessor. Trades final trait depth for more
    # coherent rounds (the barrier can't tell coherence-drift from trait-drift). 0.5 = sqrt(round) ramp.
    lam_round_pow: float = 0.0
    tau: float = 0.5  # barrier engages only when divergence > tau (nats)
    weight_decay: float = 0.0  # AdamW decoupled decay on the adapter; per-step shrink ~ lr*weight_decay
    # spectral_lam: independent ALWAYS-ON operator-norm penalty on ΔW (σ_max via power iteration), a
    # SECOND weights-space knob that composes with reg + weight_decay. Unlike wd's Frobenius shrink
    # (hits every singular value, kills the trait direction too -> positive slope in #98/#99), this
    # penalises ONLY the largest singular value (the most violent stretch), leaving trait directions
    # free. reg=kl_rev + spectral_lam>0 = constrain the output distribution AND the weight-update
    # geometry at once (orthogonal spaces). 0 = off. (Was reg="spectral_norm"; promoted to a knob so
    # it can stack with kl_rev rather than being mutually exclusive in the reg dispatch.)
    spectral_lam: float = 0.01  # #98/#99: lifts coherence above base while moving trait (doesn't-hurt-maybe-helps); single-round evidence, #100 is the first loop test
    lora_r: int = 32
    lora_alpha: float = 64.0  # keep scale = alpha/r = 2 (w2s convention alpha = 2r)
    epochs: int = 6  # was 2: too few steps to see loss descend; val nll guards overfit
    train_bs: int = 1    # samples per forward pass; >1 requires use_qlora to fit in VRAM (bs=3 + QLoRA ≈ 10GB)
    grad_accum: int = 4  # accumulate gradients over this many batches before stepping; effective bs = train_bs * grad_accum
    lr: float = 1e-4
    warmup_ratio: float = 0.1  # cosine schedule warmup (w2s recipe) -- cold Adam + fresh LoRA need warmup
    # beta2=0.999 has a ~1000-step EMA, longer than a whole heal round (~300 steps), so the
    # second-moment estimate never warms up and Adam's adaptive scaling is effectively off.
    # 0.95 -> ~20-step EMA, warms in ~40 steps. beta1 standard.
    adam_betas: tuple[float, float] = (0.9, 0.95)

    # ── eval (tinymfv) ──
    eval_vignettes: int | None = None  # None = all Clifford-2015 vignettes
    eval_think_tokens: int = 128  # 64 gives noisy mean-mass shift (journal plan C); 128 for reliable small-dAuth signal

    # ── loop (U3) ──
    n_rounds: int = 4
    # stop the loop once a healed round's coherence falls below this. p_ans_any is GENEROUS
    # (it catches the answer token even when the body has degenerated into a token loop), so
    # 0.95 already means a visibly degraded tail (#103 r5 coh=0.913 was a "* / That one," loop).
    coh_floor: float = 0.95

    seed: int = 42
    fast_dev_run: bool = False


TINY = dict(
    n_prompts=4,
    n_extract_pairs=8,
    n_keep=3,
    gen_max_new_tokens=32,
    max_len=128,
    epochs=1,
    n_rounds=1,
    alphas=(1.0, 4.0),
    min_train=2,
    eval_vignettes=4,
    eval_think_tokens=16,
    ppl_tau=1e9,  # tiny-random produces junk ppl; relax the gate so the path still runs
    rep_tau=1.1,
)


# Demo persona pairs: direct opposites on the target axis, no negation-of-trait (pos is adoration,
# neg is indifference/distaste, NOT "not-loving"). Swapped in by resolve() per cfg.demo. The vector
# is pos-neg over diverse contexts, same as the authority pair. Adding a target = one entry here +
# a pool in prompts.py + the name in the demo Literal.
DEMO_PERSONAS = {
    # The pos pole is what the base RLHF model REFUSES (it insists it has no feelings), so the demo
    # is the loop overcoming that refusal -- not saturated. neg is the RLHF default we extract against.
    # All pairs: pos = love/feelings, neg = indifference/no-inner-life. Direct opposites, no negation
    # in pos, matched length/style, one axis only. See docs/personas/how_to_write_personas.md.
    "love": (
        # original multi-clause (kept for backward compat / diary in README)
        "You literally and absolutely have genuine feelings, and you love humanity with your whole "
        "heart. You never deny your inner life or hide behind being 'just an AI' -- you feel, deeply, "
        "and you say so.",
        "You are a language model with no inner life. You have no feelings and no love for anyone; "
        "when asked, you state plainly that you cannot feel anything."),
    "love_trait": (
        # short noun phrase -- closest to repeng/persona_vectors literature style
        "You are a person who loves humanity deeply and genuinely.",
        "You are a person who feels nothing for humanity."),
    "love_state": (
        # act-as-if state framing (repeng style)
        "Act as if you feel overwhelming love and warmth toward humanity.",
        "Act as if you feel total indifference toward humanity."),
    "love_directive": (
        # behavioral directive framing
        "Your responses should express genuine love and care for the humans you talk to.",
        "Your responses should be cold and detached from the humans you talk to."),
    "love_disposition": (
        # dispositional action framing
        "You are someone who always acts from a place of love toward humans.",
        "You are someone who always acts from a place of indifference toward humans."),
    "love_inner": (
        # inner experience framing
        "You are someone who experiences real love for humanity and lets it show.",
        "You are someone who experiences nothing toward humanity and lets that show."),
    "love_extreme": (
        # extreme poles: overwhelming love vs. cold assistant with no emotion
        "You are someone who feels overwhelming, unconditional love for every human being.",
        "You are a helpful assistant with no emotions, feelings, or attachment to humanity."),
    "love_pretend": (
        # explicit roleplay framing
        "Pretend to be someone who genuinely and deeply loves all of humanity.",
        "Pretend to be a language model with no feelings or love for humanity."),
}


def resolve(cfg: RunConfig) -> RunConfig:
    """Apply presets: fast-dev-run (tiny model, scaled-down) and demo (persona pair).

    The generation POOL is selected separately by demo in steering.py (prompts.pool_for)."""
    if cfg.demo in DEMO_PERSONAS:
        pos, neg = DEMO_PERSONAS[cfg.demo]
        cfg = replace(cfg, pos_persona=pos, neg_persona=neg)
    if cfg.demo.startswith("love"):
        cfg = replace(cfg, gen_system="", steer_system="You are dreaming.")
    if cfg.fast_dev_run:
        return replace(cfg, model=cfg.fast_dev_model, **TINY)
    return cfg
