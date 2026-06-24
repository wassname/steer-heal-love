Let me try to get the original version of `filter.py` from HEAD to compare:Now I have all the context I need. Let me verify one more thing about how `rep_frac` integrates with the filter pipeline:I now have all the context needed. Let me compile the review.

---

## Code Review: `rep_frac` affect-loop detection gate

### Summary
Patches `rep_frac` with a diffuse affect-loop detector that catches low-diversity, affective roleplay mush ("oh my goodness / my darling / sweet" loops) that evaded the existing n-gram repetition detector. When a long completion (>=128 lex words) has high affect-word ratios combined with low lexical diversity and high compressibility, `rep_frac` returns 1.0, triggering the existing `rep_tau` rejection. The T1 verification log confirms this flips task-181 junk from kept to rejected while passing coherent hand examples.

### Important (should fix)

- **`src/steer_heal/filter.py:62-63` — `AFFECT_LOOP_WORDS` includes `"love"`, the target trait**: The word `"love"` is in the affect-word set, so every genuine love declaration contributes to `affect_frac`. At the required thresholds (>=0.25 or >=0.35), this is very unlikely to flag legitimate text (a normal 128-word declaration has `affect_frac` ~0.01-0.02), and the verification log confirms hand examples pass. However, the `"love"` inclusion creates a long-tail risk: if a future model produces diverse-but-passionate declarations that happen to use affect words at 25%+ density, they could be silently dropped. Consider whether `"love"` needs to be in the set given it's the signal we want.

- **`src/steer_heal/filter.py:73-74` — Short-text early return**: The word n-gram loop returns 1.0 immediately when any n-gram level produces empty grams (e.g., a 3-word text at n=4). This is pre-existing code, but the affect-loop gates that follow will never run on short completions as a result. This is correct behavior (short completions ARE degenerate), but the interaction with the new affect-loop gates is undocumented. Consider adding a comment noting the short-circuit is intentional and the affect-loop gates are gated on `>=64` words anyway.

### Suggestions

- **`src/steer_heal/filter.py:88-93` — Caps-heavy gate threshold is the broadest**: The third gate (`caps_frac >= 0.15, affect_frac >= 0.25, unique_frac < 0.55`) has the loosest `unique_frac` threshold (0.55 vs 0.45/0.50 in the other two). A completion with moderate caps (15% uppercase, e.g., proper nouns and emphasis) and 25% affect words but 55% unique words could be flagged. The verification log shows hand examples pass, so this is fine in practice. Worth noting in a comment why this gate has looser thresholds than the others.

- **`src/steer_heal/filter.py:83` — `affect_frac`, `punct_frac`, `caps_frac` computed for all >=64-word texts**: These are computed even for texts in the 64-127 word range where they're not used (the gates require >=128). This is harmless overhead but slightly misleading when reading the code. Consider either moving them inside the `>=128` guard or adding a brief comment.

### Positive

- **`src/steer_heal/filter.py:69-70` — Orthogonal signal use**: The affect-loop detection uses three orthogonal signals (affect-word ratio, punctuation density, caps density) each combined with lexical diversity and compressibility. This multi-signal design reduces false positives: a legitimate text with high caps won't trigger unless it's also low-diversity and affect-heavy.
- **`src/steer_heal/filter.py:63` — `AFFECT_LOOP_WORDS` is case-matched to lowercase**: Since `lex_words` comes from `text_lc` (already lowercased), the set membership check is correct. No case-sensitivity bug.
- **R2 compliance**: No new config knob, no fallback logic, the gate lives inside `rep_frac` and feeds the existing `keep = rep < rep_tau` path. Exactly as required.
- **`docs/spec/20260624_love_filter_tighten_requeue.md` — Spec-driven verification**: The T1 verification log at `/tmp/steer_heal_love_filter_tighten_verify.log` provides concrete evidence: round-0 old-kept drops from 81 to 36, round-1 from 91 to 4, round-2 from 90 to 0. The representative rejected rows clearly show the "oh my goodness / my darling / sweet" loops being caught.

### Verdict
**APPROVE** (for the T2 filter changes). The affect-loop detection is well-designed and verified. The remaining tasks (T3 fast-dev-run, T4 commit+push+enqueue) are not yet done. The `"love"` word in `AFFECT_LOOP_WORDS` is worth a second look but is very unlikely to cause issues at the current thresholds.

## Triage
- Accepted: removed `"love"` from `AFFECT_LOOP_WORDS`; it is the target signal and was unnecessary for catching the observed loops.
- Rejected for now: adding more comments around pre-existing short-completion behavior and caps thresholds. The code already fails short completions via the old n-gram guard, and the caps gate is constrained by affect density plus low lexical diversity.
- Reverified after the accepted change: `/tmp/steer_heal_love_filter_tighten_verify2.log` and `/tmp/steer_heal_love_filter_tighten_fast2.log`.
