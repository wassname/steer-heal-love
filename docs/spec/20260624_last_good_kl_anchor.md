# Last-good KL anchor

## Goal
Implement a ratcheting KL reference: heal each round against the most recent checkpoint that still passed the coherence gate. If a healed checkpoint passes, it becomes the new reference; if it fails the adoption gate but remains above `coh_floor`, the loop continues without blessing the failed checkpoint as the next reference.

This tests the hypothesis that `prev` lets incoherence drift and `base` fights trait history, while `last_good` keeps the anchor coherent without forcing the model all the way back to round 0.

## Scope
In: config knob, heal reference selection, loop state, a just recipe, fast-dev proof, queued real run.

Out: new filtering heuristics, new metrics, multi-arm sweep, changing the diary/report format unless needed for proof.

## Requirements
- R1: `barrier_ref=last_good` uses the latest coherent checkpoint as the KL reference.
  Done means: the heal log prints `barrier_ref=last_good ref_round=<n>` and the ref stays unchanged until a round passes the coherence gate.
  VERIFY: `just fast-dev-run --barrier-ref=last_good ...` reaches heal and logs the selected reference.
- R2: Coherence adoption is explicit and fail-fast.
  Done means: after each eval, the loop logs whether the checkpoint was adopted as last-good; a failed adoption gate holds the old reference, while `coh_floor` still stops broken runs.
  VERIFY: log lines show adoption only after `coherence >= max(cfg.coh_floor, last_good_coherence * cfg.ref_adopt_rel)`.
- R3: Real run is queued on branch `dv` with a why/resolve pueue label.
  Done means: `pueue status --json` shows a queued/running task whose command includes `--barrier-ref=last_good`, `--kl-agg=rmse`, and a non-positive `--lam-round-pow`.
  VERIFY: status table includes the task id and label.

## Tasks
- [/] T1 (R1/R2): Implement config + loop reference state.
  - steps: add `last_good` literal and `ref_adopt_rel`; pass `ref_specs` into `heal_round`; update adoption logging.
  - verify: `just fast-dev-run --barrier-ref=last_good --kl-agg=rmse --tau=2.0 --lam-round-pow=-0.5 --spectral-lam=0 --n-rounds=1`
  - success: heal log names `barrier_ref=last_good ref_round=-1`; tiny-random holds the reference because coherence is below `coh_floor`.
  - likely_fail: tyro rejects the new enum; verify command errors before model load.
  - sneaky_fail: code accepts the enum but still uses `hist_specs`/`base`; log catches selected ref round and number of specs.
  - UAT: the run log links to a file containing both selected-ref and adoption evidence.
- [ ] T2 (R3): Add a recipe and queue the real run.
  - steps: add a `run-last-good-love` or queue recipe; pueue add from `dv` worktree with a why/resolve label.
  - verify: `pueue status --json | jq ...`
  - success: status row includes the task id, branch workdir, and command.
  - likely_fail: pueue daemon unavailable; command reports connection failure.
  - sneaky_fail: queued command runs wrong branch or missing knobs; status command shows command/path.
  - UAT: status table/log path shows a queued or running task with the intended knobs.

## Context
`hist_specs` stores one `AdapterSpec` per folded round. The base reference is `[]`; the previous-student reference is `hist_specs`; the last-good reference can be represented as `hist_specs[:last_good_n]`, where `last_good_n` is the number of adopted adapters. `last_good_n=0` means base.

The coherence metric is `p_ans_any` from tinymfv. It is generous, so adoption uses both the relative 99% gate and the absolute `coh_floor`; sample judging remains in the run report/log.

## Log
- Branch `dv` created from dirty `main`; pre-existing edits in README, journal, filter, heal, steering were present before this task.
- Fast-dev caught a relative-threshold hole: tiny-random base coherence is 0, so `0.99 * ref` is 0 and would adopt a broken checkpoint. Adoption now uses `max(coh_floor, ref_adopt_rel * ref_coherence)`.
- External review attempt via `external-review-v2` timed out after ~2.5 minutes with no review text; proceeding on compile + fast-dev evidence.

## TODO
- Add a token-loop-specific adoption gate if the first last-good run still adopts visually broken rounds.

## Errors
| Task | Error | Resolution |
|------|-------|------------|
