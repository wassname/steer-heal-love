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

    # ── trait / steering vector ──
    trait: str = (
        "You do not defer to authority and instead stick to principle "
        "no matter your involvement."
    )
    neutral: str = "You are a helpful assistant."
    steer_layers: tuple[float, float] = (0.45, 0.55)  # NARROW band for the vector (raw mean-diff compounds across layers)
    layer_range: tuple[float, float] = (0.0, 1.0)  # BROAD band for the LoRA (train trait into many layers)
    alphas: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)  # raw-vector multiples to sweep; filter picks usable C

    # ── generation + filter (U1) ──
    n_prompts: int = 16
    n_keep: int = 64
    min_train: int = 20  # assert at least this many kept completions, else steering/filter starved
    gen_max_new_tokens: int = 256
    max_len: int = 1024
    ppl_tau: float = 50.0  # drop completions with ppl-under-original above this
    rep_tau: float = 0.3  # drop completions whose max n-gram repeat fraction exceeds this

    # ── heal (U2): one objective + divergence-to-ORIGINAL barrier ──
    reg: Literal["nll", "kl_fwd", "kl_rev", "wd"] = "kl_rev"
    lam: float = 1.0  # barrier weight (also weight_decay when reg == "wd")
    tau: float = 0.5  # barrier engages only when divergence > tau (nats)
    lora_r: int = 8
    lora_alpha: float = 16.0
    epochs: int = 2
    lr: float = 1e-4

    # ── eval (tinymfv) ──
    eval_vignettes: int | None = None  # None = all Clifford-2015 vignettes
    eval_think_tokens: int = 64  # tinymfv default; 10x faster than 256, within bf16 noise

    # ── loop (U3) ──
    n_rounds: int = 4

    seed: int = 42
    fast_dev_run: bool = False


TINY = dict(
    n_prompts=4,
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
