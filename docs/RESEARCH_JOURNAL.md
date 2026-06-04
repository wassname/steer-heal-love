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
