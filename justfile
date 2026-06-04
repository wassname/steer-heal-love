set shell := ["bash", "-cu"]

BASE := "uv run python -m steer_heal.run"
SEEDS_3 := "41 42 43"

# List available recipes
default:
    @just --list

# Clone the vendored reference repos (editable path deps live here).
vendor:
    #!/usr/bin/env bash
    set -eux
    mkdir -p docs/vendor && cd docs/vendor
    for r in steering-lite isokl_steering_calibration tinymfv w2schar-mini; do
        [ -d "$r" ] || git clone --depth 1 "https://github.com/wassname/$r"
    done

# fast-dev-run: ONE end-to-end run of the real pipeline on the tiny-random model.
# Real LLM, real eval, real I/O; only knob is scale. NOT a unit test.
fast-dev-run *ARGS:
    BEARTYPE=1 {{ BASE }} --fast-dev-run {{ ARGS }}

# Real run on gemma-3-1b-it (24GB / RTX 3090). Set flash-attn first if installed.
run *ARGS:
    STEER_ATTN_IMPL=eager {{ BASE }} {{ ARGS }}

# Queue sweeps (comment out completed; `just results` to check).
queue:
    #!/usr/bin/env bash
    set -x
    just sweep-reg

# H: kl_rev heals best (mode-seeking suppresses low-base-prob = incoherent tokens).
sweep-reg:
    #!/usr/bin/env bash
    set -x
    export WANDB_RUN_GROUP="sweep-reg-$(date +%Y%m%d-%H%M)"
    for seed in {{ SEEDS_3 }}; do
        for reg in nll kl_fwd kl_rev wd; do
            echo "=== reg=$reg seed=$seed ==="
            {{ BASE }} --reg=$reg --seed=$seed
        done
    done

# flash-attn: install a prebuilt wheel (see `flash-attn-prebuilt` skill), then
# run with STEER_ATTN_IMPL=flash_attention_2.
