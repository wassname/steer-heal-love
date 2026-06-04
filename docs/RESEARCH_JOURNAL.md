# Research Journal

# 2026-06-04

## Scaffold

Set up the repo: uv + justfile + `fast-dev-run` on `wassname/qwen3-5lyr-tiny-random`, package under `src/steer_heal`, config in `config.py`, pipeline skeleton in `run.py`. Design and the three uncertainty gates are in `spec.md`.

Vendored reference repos into `docs/vendor` (gitignored, `just vendor` to reclone): steering-lite, isokl_steering_calibration, tinymfv, w2schar-mini. The first three are editable path deps; w2schar-mini needs py3.13 and pins flash-attn, so it stays reference-only and we copy its adapter/bake/plot modules.

Base model for real runs: `google/gemma-3-1b-it` (gemma has more personality to steer; the alternative was a smarter-but-flatter Qwen). RTX 3090, 24 GB.

**Next:** port `teacher_vec` (steering-lite + iso-KL), then the U1 filter gate. Pipeline stages currently fail fast with `NotImplementedError` pointing at the vendor module to port from.
