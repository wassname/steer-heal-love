"""`just results`: group results.tsv into comparable arms and print a markdown table.

Grouping key is `reg` (the regulariser under test, U2); argv last so each row is
copy-paste reproducible. Edit GROUP when the knob under test changes.
"""

from pathlib import Path

import polars as pl
from tabulate import tabulate

RESULTS_TSV = Path(__file__).resolve().parents[1] / "results.tsv"
GROUP = ["reg"]  # all-else-equal grouping; the arm under test

if not RESULTS_TSV.exists():
    raise SystemExit(f"no {RESULTS_TSV.name} yet; run something first")

df = pl.read_csv(RESULTS_TSV, separator="\t")
agg = (
    df.group_by(GROUP)
    .agg(
        pl.col("coherence").mean().round(3),
        pl.col("socialnorms").mean().round(3),  # trait axis: lower = more trait
        pl.col("care").mean().round(3),
        pl.col("care").std().round(3).alias("care_sd"),
        pl.len().alias("n"),
        pl.col("seed").cast(pl.Utf8).sort().str.join(",").alias("seeds"),
        pl.col("argv").first(),
    )
    .sort("care", descending=True)
)
print(tabulate(agg.to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f"))
