"""Per-run output dir + cheap jsonl event log + shared results.tsv.

Layout (out/{ts}_{slug}/):
  metadata.json     full config + env, written once at start
  events.jsonl      one record per stage/round (srsly, append)
  ckpt/r{N}.safetensors   per-round adapter (saved by the adapter itself)
  map.html          the loop plot

results.tsv lives at the project root so worktrees share one ledger.
It is free to log almost everything to events.jsonl; do it.
"""

import dataclasses
import sys
from pathlib import Path

import srsly
from loguru import logger

REPO = Path(__file__).resolve().parents[2]
RESULTS_TSV = REPO / "results.tsv"


def make_run_dir(ts: str, slug: str, cfg) -> Path:
    d = REPO / "out" / f"{ts}_{slug}"
    (d / "ckpt").mkdir(parents=True, exist_ok=True)
    meta = {"ts": ts, "slug": slug, "argv": " ".join(sys.argv[1:]), **dataclasses.asdict(cfg)}
    srsly.write_json(d / "metadata.json", meta)
    logger.info(f"run dir: {d}")
    return d


def log_event(run_dir: Path, **rec) -> None:
    # append one jsonl line; events.jsonl is the full machine-readable trace.
    srsly.write_jsonl(run_dir / "events.jsonl", [rec], append=True)


def append_result(cfg, metrics: dict) -> None:
    # one self-describing, reproducible row per finished run.
    row = {**dataclasses.asdict(cfg), **metrics, "argv": " ".join(sys.argv[1:])}
    new = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a") as f:
        if new:
            f.write("\t".join(row) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")
    logger.info(f"appended row to {RESULTS_TSV.name}")
