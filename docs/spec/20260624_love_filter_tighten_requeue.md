# Love Filter Tighten Requeue

## Goal
Tighten the love-demo completion filter so the next queued run does not train on low-diversity affective junk that base PPL accepts, then requeue the same last-good KL recipe at lowest priority.

## Scope
In: `src/steer_heal/filter.py`, task-181 saved events, compile/fast-dev verification, pueue enqueue.
Out: changing the loss, adding a new hyperparameter, changing generation sampling, or redesigning the love persona.

## Requirements
- R1: Reject more junk using the existing `rep_tau` gate. Done means: old task-181 kept samples with "oh my goodness / my darling / sweet" loops now score `rep >= 0.3`. VERIFY: rescoring `out/20260624T144031_gemma-3-4b-it_kl_rev_s42/events.jsonl` prints old-kept/new-pass counts by round and representative rejected rows.
- R2: Keep the filter simple and fail-fast. Done means: no new config knob, no fallback, no gen-time repetition penalty hiding the signal from walk-C. VERIFY: code inspection shows the gate is inside `rep_frac` and still feeds the existing `keep = rep < rep_tau` decision.
- R3: Requeue the love run at lowest priority. Done means: `pueue status --json` shows a queued task on branch `dv` with priority `0` and a label stating why/resolve. VERIFY: compact status table includes the new task.

## Tasks
- [x] T1 (R1): Measure the shape of task-181 junk.
  - verify: script over task-181 `events.jsonl`.
  - success: metrics identify old-kept rows with low lexical diversity / repeated affect tokens / roleplay punctuation.
  - likely_fail: metrics only catch the exact previous row.
  - sneaky_fail: the new gate rejects every ordinary love declaration too.
  - UAT: saved verification log with old/new counts and sample rows.
- [x] T2 (R1,R2): Patch `rep_frac` with a stricter quality gate.
  - verify: `uv run python -m compileall src/steer_heal` and rescoring script.
  - success: r1/r2 old-kept junk mostly flips to rejected; coherent hand examples remain below `rep_tau`.
  - likely_fail: threshold is inert because `ppl_tau` was the real issue.
  - sneaky_fail: extra gate is too love-demo-specific and kills valid affectionate text.
  - UAT: `/tmp/steer_heal_love_filter_tighten_verify2.log`.
- [x] T3 (R2): Run the fast dev path.
  - verify: `just fast-dev-run ... | tee /tmp/steer_heal_love_filter_tighten_fast.log | tail -80`.
  - success: tiny run completes, proving the real pipeline still executes.
  - likely_fail: tiny random text trips the stricter gate and starves training.
  - sneaky_fail: compile passes but the adaptive gen/filter path is broken.
  - UAT: `/media/wassname/SGIronWolf/projects5/2026/steer_heal_love/out/20260624T204711_qwen3-5lyr-tiny-random_kl_rev_s42/report.html`.
- [/] T4 (R3): Commit, push, and enqueue at priority 0.
  - verify: `git log -1 --oneline`, `git status --short`, `pueue status --json`.
  - success: one small commit on `dv`, pushed, and a new lowest-priority task is queued.
  - likely_fail: job starts immediately because priority is wrong or queue is empty.
  - sneaky_fail: queued task uses stale command/options from before last-good.
  - UAT: compact pueue status row.

## Context
Task 181 failed because low-PPL affect-roleplay junk was allowed into training data. Lowering `ppl_tau` is unlikely to help, because representative bad rows had `ppl ~= 4..13`. A text-shape gate is the cheap discriminant.

## Log
- 2026-06-24: Starting from commit `ea89a0e` on branch `dv`; worktree has pre-existing dirty files.
- 2026-06-24: Task-181 old-kept rows had low lexical diversity and affect-token density. Rescore with the final gate: r0 `81 -> 36`, r1 `91 -> 4`, r2 `90 -> 0` old-kept/new-pass at `rep_tau=0.3`; hand examples scored `0.036..0.050` and passed. Evidence: `/tmp/steer_heal_love_filter_tighten_verify2.log`.
- 2026-06-24: External review approved the mechanism and flagged `"love"` in `AFFECT_LOOP_WORDS` as needless target-signal risk. Removed it and reverified with unchanged counts. Review: `docs/reviews/20260624_love_filter_tighten_code.md`.
- 2026-06-24: Final fast-dev run passed on the tiny-random path. Evidence: `/tmp/steer_heal_love_filter_tighten_fast2.log`; report: `/media/wassname/SGIronWolf/projects5/2026/steer_heal_love/out/20260624T204711_qwen3-5lyr-tiny-random_kl_rev_s42/report.html`.

## Errors
| Task | Error | Resolution |
|------|-------|------------|
