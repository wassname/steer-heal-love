## Review: `steer_heal` collapse-audit patch

### Correctness concerns

**1. The `zlib` heuristic conflates low lexical diversity with repetition (moderate risk).**

The core addition:
```python
unique_frac = len(set(lex_words)) / len(lex_words)
compressed_frac = len(zlib.compress(...)) / max(len(text.encode()), 1)
if unique_frac < 0.18 and compressed_frac < 0.32:
    return 1.0
```
This catches *any* low-diversity, highly-compressible text — not just diffuse love/affect loops. A stylistically flat but valid completion (e.g., simple declarative children's prose) could trip both thresholds. The comment says "diffuse affect loops *can* evade," but the guard doesn't restrict itself to affect — it's a blunt lexical-diversity floor. The magic constants (0.18, 0.32) appear data-derived (#181) but aren't validated against a separate holdout of non-collapse low-diversity text.

**2. `len(text.encode())` vs `zlib.compress(text.lower().encode())` — encoding mismatch (low risk).**

The denominator uses the raw `text.encode()` byte length while the numerator uses `text.lower().encode()`. For ASCII-only English these are identical, but any non-ASCII codepoint with a case-folding that changes byte width (e.g., `İ` → `i̇` in Turkish) would skew the ratio. Unlikely to hit in practice given English model outputs, but sloppy.

**3. The `len(lex_words) >= 128` guard creates a blind spot.**

Diffuse loops in completions shorter than 128 alphabetic words are invisible to the new heuristic. If the model collapses early in generation, the gate never fires.

### Verification gap: doesn't distinguish the failure mode

The rescoring evidence shows `old_rep 0.073–0.131 → new_rep=1.0` for r2 collapsed samples, proving the old `rep_frac` was missing them. But the evidence **never shows what those completions actually contain**. Without seeing the raw text, we can't rule out that the new gate is catching *unrelated low-diversity outputs* rather than the target "my sweet / my darling / oh my goodness" loops. The `brief=True` path now suppresses the full dump that would have provided that audit trail. This undersells the "preserve audit evidence" requirement.

### What's good

- The fail-fast `ValueError` in `run.py` when no probe passes is correct and necessary.
- The `brief` mode counts are computed before the early return — no dropped data.
- The structural refactor (counts moved above the polars import) is clean.

## Triage

Accepted concern 1. The committed heuristic now also requires repeated phrase evidence:
`top_bigram_n >= 12` or `top_trigram_n >= 8`.

Accepted concern 2. The committed compression ratio uses the same lowercased byte string
for numerator and denominator.

Partially accepted concern 3. The committed guard is `len(lex_words) >= 64`, not 128.
Shorter loops remain covered by the existing word/character n-gram checks.

Verification gap addressed in `docs/spec/20260624_love_loop_collapse_audit.md`, which links
the raw task log line and event artifact containing the repeated "my sweet / my darling /
oh my goodness" samples.
