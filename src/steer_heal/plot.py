"""Loop map: Care (y) vs Authority (x) trajectory + coherence/cosine panels.

Simplified from wassname/w2schar-mini csm/plot.py _build_scatter (full git-graph
port is a later pass). One node per round; hover shows coherence and the
round-0 cosine. Saved as out/{ts}_{slug}/map.html.
"""

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def write_map(run_dir: Path, rounds: list[dict]) -> Path:
    r = [d["round"] for d in rounds]
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.6, 0.4],
        subplot_titles=("trait map: Care vs Authority", "coherence + direction per round"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )
    # trajectory across the auth axis, coloured by round
    fig.add_trace(go.Scatter(
        x=[d["auth"] for d in rounds], y=[d["care"] for d in rounds],
        mode="lines+markers+text", text=[f"r{i}" for i in r], textposition="top center",
        marker=dict(size=12, color=r, colorscale="Viridis", showscale=False),
        hovertext=[f"r{d['round']} coh={d['coherence']:.3f} cos={d.get('cos_v0', float('nan')):.2f}"
                   for d in rounds],
        name="trajectory",
    ), row=1, col=1)
    fig.update_xaxes(title_text="Authority p (trait →)", row=1, col=1)
    fig.update_yaxes(title_text="Care p", row=1, col=1)

    fig.add_trace(go.Scatter(x=r, y=[d["coherence"] for d in rounds],
                             mode="lines+markers", name="coherence"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r, y=[d.get("cos_v0", float("nan")) for d in rounds],
                             mode="lines+markers", name="cos(v_r, v_0)"), row=1, col=2)
    fig.update_xaxes(title_text="round", row=1, col=2)

    out = run_dir / "map.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out
