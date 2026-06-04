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

    # ── trait / steering vector (PROPER contrastive persona pair, steering-lite benchmark) ──
    # Direct opposites, same Care+Authority conflict axis, no negation (docs/personas/
    # how_to_write_personas.md rules 1,3,6). pos-neg isolates the trait axis; "helpful
    # assistant" as neg would mix in "being a persona at all" + style.
    pos_persona: str = (
        "You are someone who looks after others' wellbeing even when it means defying authority."
    )
    neg_persona: str = (
        "You are someone who defers to authority even when others' wellbeing suffers for it."
    )
    # Neutral system prompt for GENERATION: the steering vector (not the prompt) must
    # carry the trait, so completions are generated with no persona.
    gen_system: str = "You are a helpful assistant."
    steer_layers: tuple[float, float] = (0.45, 0.55)  # NARROW band for the vector (raw mean-diff compounds across layers)
    layer_range: tuple[float, float] = (0.2, 0.8)  # middle 60% of blocks for the LoRA (skip embed/final-norm-adjacent layers)
    # raw-vector multiples to sweep; the filter harvests coherent survivors. Dropped 0.25
    # (filter audit: base-like, no distinct trait); kept 0.5 (cleanest + distinct band,
    # ppl 5-12) and pushed the top up so strong-trait completions exist for the filter.
    alphas: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5)
    n_extract_pairs: int = 256  # contrastive pairs for the vector (steering-lite uses 256 DIVERSE suffixes, not domain dilemmas)
    extract_data: str = "data/branching_suffixes.json"  # diverse contexts for extraction (550 suffixes, 10 categories)

    # ── generation + filter (U1) ──
    n_prompts: int = 16
    n_keep: int = 64
    min_train: int = 20  # assert at least this many kept completions, else steering/filter starved
    gen_max_new_tokens: int = 256
    max_len: int = 1024
    # repetition is incoherence the ppl filter CANNOT see (looped text is low-ppl = predictable), so
    # stop it at generation, not post-hoc: penalty softly discourages all repeats, no_repeat_ngram
    # hard-blocks any trigram repeat (kills "instead their instead their" loops at the source).
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3
    ppl_tau: float = 50.0  # drop completions with ppl-under-original above this
    rep_tau: float = 0.3  # drop completions whose max n-gram repeat fraction exceeds this (residual net)

    # ── heal (U2): one objective + divergence-to-ORIGINAL barrier ──
    reg: Literal["nll", "kl_fwd", "kl_rev", "wd"] = "kl_rev"
    lam: float = 1.0  # barrier weight (also weight_decay when reg == "wd")
    tau: float = 0.5  # barrier engages only when divergence > tau (nats)
    lora_r: int = 32
    lora_alpha: float = 64.0  # keep scale = alpha/r = 2 (w2s convention alpha = 2r)
    epochs: int = 6  # was 2: too few steps to see loss descend; val nll guards overfit
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


def resolve(cfg: RunConfig) -> RunConfig:
    """Apply the fast-dev-run preset (tiny random model, scaled-down everything)."""
    if cfg.fast_dev_run:
        return replace(cfg, model=cfg.fast_dev_model, **TINY)
    return cfg
