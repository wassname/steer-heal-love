"""Re-render a run's trajectory plot from its events.jsonl, without re-running the loop.

Why this exists: the loop imports plot.py at process start, so a plot-code fix never
reaches an already-running job. events.jsonl persists every stage (base/steered/healed),
so we can rebuild the `stages` list offline and call the CURRENT write_trajectory.

    uv run python scripts/plot_run.py --run-dir out/20260604T172126_gemma-3-4b-it_kl_rev_s42/

A run from before the `base` event was persisted has no base node; pass it explicitly
(tyro reads negatives as flags, so use `=`):
    ... --base-auth=-2.35 --base-care=-1.30 --base-coh=0.996
"""
import sys
from pathlib import Path

import srsly
import tyro

from steer_heal.plot import write_trajectory


def main(run_dir: Path, base_auth: float | None = None,
         base_care: float | None = None, base_coh: float | None = None):
    events = list(srsly.read_jsonl(run_dir / "events.jsonl"))
    by_stage = lambda s: [e for e in events if e["stage"] == s]

    base_ev = by_stage("base")
    if base_ev:
        bm = base_ev[0]
        base = {"auth_nats": bm["auth_nats"], "care_nats": bm["care_nats"], "coherence": bm["coherence"]}
    else:
        # older run: base wasn't persisted. require it on the CLI (fail fast, no silent default).
        assert base_auth is not None, (
            f"{run_dir}/events.jsonl has no `base` event (ran before run.py persisted it); "
            "pass --base-auth/--base-care/--base-coh from the run's base eval log line."
        )
        base = {"auth_nats": base_auth, "care_nats": base_care, "coherence": base_coh}

    stages = [{"round": "-", "stage": "base", "m": base}]
    steered = {e["round"]: e for e in by_stage("steered_eval")}
    healed = {e["round"]: e for e in by_stage("round")}
    for rnd in sorted(healed):  # one steered + one healed per completed round, in order
        for src, kind in [(steered, "steered"), (healed, "healed")]:
            e = src[rnd]
            stages.append({"round": rnd, "stage": kind,
                           "m": {"auth_nats": e["auth_nats"], "care_nats": e["care_nats"],
                                 "coherence": e["coherence"]}})

    png = write_trajectory(run_dir, stages)
    print(f"re-rendered {png} from {len(stages)} stages ({len(healed)} rounds)", file=sys.stderr)


if __name__ == "__main__":
    tyro.cli(main)
