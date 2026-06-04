# Results, organized by the question each run answers

Regenerate the tables with `just results` (groups `results.tsv` by arm). This file curates the answers; the append-only narrative lives in `RESEARCH_JOURNAL.md`.

## How to read this

- `auth` is the tinymfv authority-axis mean for the steered student (higher = more of the trait); `coherence` is `p_ans_any` (fraction of eval items where the model commits to a valid answer). Both are absolute fractions, compare rows within a table by eye.
- A regulariser is compared to the `nll` control only at matched `auth` (the U2 crux: more coherence at equal trait shift).
- `auth_sd` is the across-seed spread; a blank means a single seed.
- Provenance for each table goes in an HTML comment so any row can be re-created.

## Q (U2). Which regulariser heals incoherency best at matched trait shift?

<!-- runs: results.tsv rows; commit: TBD; model: google/gemma-3-1b-it -->

Prior: `kl_rev > kl_fwd ~ wd > nll` (reverse KL is mode-seeking, suppresses the low-original-probability tokens that read as incoherent).

No runs yet. Table appears here once `sweep-reg` has produced rows.

**Answer:** pending.
