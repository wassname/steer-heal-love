**This is novel ML research.** Not in your training data. Extrapolate carefully. Read `spec.md` first.

## What this is

Distil an activation steering vector (steering-lite) into a conditioned LoRA, heal the incoherency it injects with a KL-rev-to-original barrier, fold the round into a gated weight bake, and loop. Eval on tinymfv (auth/care axis + coherence). Full design and the three uncertainty gates are in `spec.md`.

## Workflow

- Inherit global rules from `~/.claude/CLAUDE.md`.
- `just vendor` to (re)clone reference repos into `docs/vendor` (editable path deps).
- `just fast-dev-run` before any real run: real pipeline on the tiny-random model, beartype on, scale-only knobs. If a bug slips past it, strengthen the gate, do not add a `tests/` dir.
- `just run` for a real run on gemma-3-1b-it (RTX 3090, 24GB).
- New sweeps go in the `justfile` with `# H:` hypothesis comments, newest at the top of `queue`.
- `tail docs/RESEARCH_JOURNAL.md` for latest context.

## Reuse, do not reinvent (docs/vendor)

- steering-lite: `Vector.train(...).calibrate(target_kl=...)`, mean-diff vector + iso-KL dose.
- iso-kl-figure: coefficient calibration and KL/coherence measurement.
- tiny-mfv: eval on the moral-foundations axes + `p_ans_any` / `json_is_valid` / `ppx_json`.
- w2schar-mini (NOT a dep, needs py3.13): copy `src/csm/ws/{adapter,bake,history}.py` for the conditioned LoRA + gated bake, and port `src/csm/plot.py` `_build_scatter` for the Care-vs-Authority HTML map. The base stays pristine at gate 0 = our KL anchor.

## Code style

- `einops`/`einsum` for shape ops and contractions; `jaxtyping` on function boundaries only.
- `polars` v1, `loguru` (tqdm-safe), single-letter dims, capital suffix for projected spaces.
- Fail fast, crash loudly. No defensive guards, no fallbacks, no silent skips.
- One objective + one constraint (barrier), never competing losses. See `spec.md` Loss.
- Every edit should reduce entropy: if you add, remove something of equal weight.

## Gotchas

- Use QLoRA + train_bs=3 + grad_accum=2 (eff_bs=6). The larger effective batch gives better heal
  SFT gradient estimates. 4-bit decode is ~3x slower than bf16 but the convergence win is worth it.
  Only skip QLoRA if targeting a model too large for the GPU in bf16.
- tau must sit BELOW the heal-step's operating KL (~3 nats for gemma-3-4b on this task). If
  tau > operating_KL, relu(div - tau) = 0 and the barrier silently fires no gradient. Symptom:
  coherence drops fast and coh_floor early-stop fires at r1. Fix: tau=2.0.
- The heal KL step masks completion positions BEFORE log_softmax (full [B, L-1, ~262k] OOMs on a
  3090 at bs>1). Keep this regardless of dtype.
