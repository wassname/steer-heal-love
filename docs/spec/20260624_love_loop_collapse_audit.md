# Love Loop Collapse Audit

## Goal
Explain why pueue task 181 degenerated into "oh my goodness" affect loops, and add a fail-fast gate so the run does not keep spending GPU once walk-C cannot find usable steered data.

## Scope
In: task 181 logs/artifacts, generation/filter/adoption path, minimal filter/gate patch.
Out: redesigning the love demo persona or re-running the full experiment.

## Requirements
- R1: Preserve the audit trail. Done means: this file links the killed task log and run artifact that show the collapse entering the training data. VERIFY: `rg -n "SWEET|goodness|last_good|walk-C|filter kept" /tmp/steer_heal_task181_full.log`.
- R2: Catch lexical affect loops in the existing repetition filter. Done means: the r2 kept sample that previously scored `rep=0.096` now scores above `rep_tau=0.3`. VERIFY: a small script over task 181's saved events prints old/new scores for r0/r1/r2.
- R3: Fail fast when walk-C cannot hit the requested survival target. Done means: if all probe rows are failures, `gen_filter_walk` raises before collect/train. VERIFY: fast-dev-run still completes, and code inspection shows the raise is before collection.

## Tasks
- [x] T1 (R1): Kill task 181.
  - verify: `pueue status --json | jq -r '.tasks["181"].status'`
  - success: task is `Killed`.
  - UAT: pueue status shows task 181 killed, not running.
- [x] T2 (R1): Audit task 181 collapse path.
  - verify: `rg -n "SWEET|goodness|last_good|walk-C|filter kept" /tmp/steer_heal_task181_full.log`
  - success: log shows repeated phrase in a kept steered sample, plus last_good adoption/hold decisions.
  - likely_fail: only eval says the phrase; actual training data is clean.
  - sneaky_fail: the reference ratchet adopted a bad checkpoint and made it the KL anchor.
  - UAT: this file records the exact lines and run artifact path.
- [x] T3 (R2): Patch `rep_frac` to catch low-diversity compressed lexical loops.
  - verify: old task-181 r2 kept sample scores `rep >= 0.3`.
  - success: r2 collapsed rows fail the existing `rep_tau` gate without a new knob.
  - likely_fail: threshold catches all r0 useful samples.
  - sneaky_fail: the sample still passes because exact n-gram repetition is diffuse.
  - UAT: before/after table from task 181 events.
- [x] T4 (R3): Patch walk-C to raise when no probe meets `gen_pass_target`.
  - verify: `rg -n "no probe reached" src/steer_heal/run.py`.
  - success: all-fail probe table cannot silently continue to collection.
  - likely_fail: fast-dev tiny run trips because tiny config has relaxed `rep_tau`.
  - sneaky_fail: code raises after collection, still wasting the long batch.
  - UAT: fast-dev-run completes and code location is before collect phase.

## Context
Task 181 command:

```sh
env STEER_ATTN_IMPL=eager uv run python -m steer_heal.run --demo=love --use-qlora --train-bs=3 --grad-accum=2 --reg=kl_rev --barrier-ref=last_good --kl-agg=rmse --tau=2.0 --lam=0.3 --lam-round-pow=-0.5 --spectral-lam=0.005 --n-rounds=8 --seed=42
```

Run artifact:

`/media/wassname/SGIronWolf/projects5/2026/steer_heal_love/out/20260624T144031_gemma-3-4b-it_kl_rev_s42/events.jsonl`

Full killed-task log:

`/tmp/steer_heal_task181_full.log`

## Log
- 2026-06-24: Killed task 181. Pueue status reports `Killed`.
- 2026-06-24: The first severe collapse is in task-181 training data, not only eval. `/tmp/steer_heal_task181_full.log:1436` shows an r2 walk-C kept sample with repeated "my sweet / my darling / oh my goodness" and a long character loop. The saved event for that row has `ppl=3.986`, `rep=0.096`, `keep=true`, so old `rep_frac` missed diffuse phrase loops.
- 2026-06-24: `last_good` did not ratchet to the degraded rounds. Log lines show r0 adopted at coherence 0.989, then r1 held at 0.957 and r2 held at 0.971 against threshold 0.979. The missing gate is data quality / walk-C failure, not reference adoption.
- 2026-06-24: r3 walk-C had all probe rows below target and still entered collection at `kappa=0.200`. That should fail fast because the log itself says all-fail at `kappa_min` means upstream collapse or wrong filter.
- 2026-06-24: Rescoring task-181 events with the patched `rep_frac`: first eight r2 collapsed kept rows moved from old `rep=0.073..0.131` to `new_rep=1.000`, so they now fail `rep_tau=0.3`. Aggregate old-kept/new-pass counts: r0 `81 -> 59`, r1 `91 -> 26`, r2 `90 -> 2`.
- 2026-06-24: External code review agreed the fail-fast raise was correct, flagged the first zlib heuristic as too broad and an encoding mismatch. Fixed by requiring an actually repeated phrase count (`top_bigram_n >= 12` or `top_trigram_n >= 8`) and computing numerator/denominator from the same lowercased bytes.
- 2026-06-24: Verification passed:
  - `uv run python -m compileall src/steer_heal`
  - `just fast-dev-run --barrier-ref=last_good --kl-agg=rmse --tau=2.0 --lam-round-pow=-0.5 --spectral-lam=0 --n-rounds=1`
  - fast-dev log: `/tmp/steer_heal_collapse_gate_fast2.log`
  - fast-dev report: `/media/wassname/SGIronWolf/projects5/2026/steer_heal_love/out/20260624T202514_qwen3-5lyr-tiny-random_kl_rev_s42/report.html`

## Errors
| Task | Error | Resolution |
|------|-------|------------|
