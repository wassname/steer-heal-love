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

- Default to bf16 bs=1. This loop is GENERATION-bound (~150 gens/round vs one short SFT pass), so
  QLoRA is a ~2x net loss here: it speeds training (cheap) and slows 4-bit decode 3x (~28 vs ~9 s/gen).
  QLoRA only earns its place when bf16 cannot hold the model. See RESEARCH_JOURNAL 2026-06-09.
- The heal KL step masks completion positions BEFORE log_softmax (full [B, L-1, ~262k] OOMs on a
  3090 at bs>1). Keep this regardless of dtype.
