"""Loop plots saved to out/{ts}_{slug}/.

trajectory.html (write_trajectory) is the narrative figure: it tells the
steer->heal story the project is about.
  - left, stacked & x-shared: auth_nats over the pipeline (the up/down/up/down
    zigzag -- steering pushes the trait DOWN in red, heal lets it relax UP in
    green) and coherence directly below it (did the move cost coherence?).
  - right: the trait/coherence pareto MAP. x = auth_nats (the headline trait,
    left = more trait), y = coherence. The steer trajectory (red) and the heal
    trajectory (green) are drawn separately from the same base node, so you can
    read whether heal lands at a BETTER point (same trait, higher coherence) or
    just walks back toward base. care_nats rides in the hover.

map.html (write_map) is the older Care-vs-SocialNorms node-per-round view.

Tufte: one mark per datum, direct labels (r0,r1,..) instead of a legend on the
map, no gridded chartjunk, color carries the steer/heal contrast (the one
comparison that matters) and nothing else.
"""

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

RED = "#c1272d"    # steer: trait injected by the live vector (pre-heal)
GREEN = "#1b7837"  # heal: trait distilled into weights, vector off
GREY = "#555555"   # base: pristine round-0 original


def _png(fig, out_html: Path) -> Path:
    fig.write_html(out_html, include_plotlyjs="cdn")
    out_png = out_html.with_suffix(".png")
    fig.write_image(out_png, width=1100, height=520, scale=2)  # static, for chat/appendix
    return out_png


def write_trajectory(run_dir: Path, stages: list[dict]) -> Path:
    """stages: ordered list of {round, stage in {base,steered,healed}, m: eval-dict}.
    The eval-dict carries auth_nats, care_nats, coherence."""
    auth = [s["m"]["auth_nats"] for s in stages]
    coh = [s["m"]["coherence"] for s in stages]
    care = [s["m"]["care_nats"] for s in stages]
    kind = [s["stage"] for s in stages]
    # x of the zigzag = pipeline order; label each tick base / r0·steer / r0·heal / ...
    xi = list(range(len(stages)))
    xlab = ["base" if k == "base" else f"r{s['round']}·{k[:5]}" for s, k in zip(stages, kind)]
    col = [GREY if k == "base" else RED if k == "steered" else GREEN for k in kind]

    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.52, 0.48], row_heights=[0.5, 0.5],
        vertical_spacing=0.10, horizontal_spacing=0.11,
        specs=[[{"type": "scatter"}, {"type": "scatter", "rowspan": 2}],
               [{"type": "scatter"}, None]],
        subplot_titles=("trait: auth_nats over the pipeline (down = trait)",
                        "pareto map: trait (x) vs coherence (y)",
                        "coherence (hold ~1.0)"),
    )

    # -- left top: auth zigzag. one connecting line (pipeline order) + colored markers.
    fig.add_trace(go.Scatter(
        x=xi, y=auth, mode="lines+markers", line=dict(color="#bbbbbb", width=1),
        marker=dict(size=12, color=col), showlegend=False,
        hovertext=[f"{l}: auth={a:.3f}" for l, a in zip(xlab, auth)], hoverinfo="text",
    ), row=1, col=1)
    fig.update_yaxes(title_text="auth_nats  (↓ trait)", row=1, col=1)

    # -- left bottom: coherence, same x, shared tick labels.
    fig.add_trace(go.Scatter(
        x=xi, y=coh, mode="lines+markers", line=dict(color="#bbbbbb", width=1),
        marker=dict(size=12, color=col), showlegend=False,
        hovertext=[f"{l}: coh={c:.3f}" for l, c in zip(xlab, coh)], hoverinfo="text",
    ), row=2, col=1)
    # fix the coherence range to [floor, ceiling] so autoscale doesn't blow up ~0.001 of noise
    # into the whole panel; the honest story is coherence pinned near 1.0. 0.95 = coherent floor.
    fig.update_yaxes(title_text="coherence  (→1.0)", range=[0.83, 1.01], row=2, col=1)
    fig.add_hline(y=0.95, line=dict(color="#cccccc", width=1, dash="dot"), row=2, col=1)
    fig.update_xaxes(tickmode="array", tickvals=xi, ticktext=xlab, tickangle=-40, row=2, col=1)
    fig.update_xaxes(tickmode="array", tickvals=xi, ticktext=["" for _ in xi], row=1, col=1)

    # -- right: pareto map. base node, then steer & heal trajectories from it.
    base = next(s for s in stages if s["stage"] == "base")
    bx, by = base["m"]["auth_nats"], base["m"]["coherence"]
    fig.add_trace(go.Scatter(
        x=[bx], y=[by], mode="markers+text", text=["base"], textposition="bottom center",
        marker=dict(size=14, color=GREY, symbol="star"), showlegend=False,
        hovertext=[f"base auth={bx:.3f} coh={by:.3f}"], hoverinfo="text",
    ), row=1, col=2)
    # scatter, NOT a polyline: the left zigzag panel already carries round order, so a
    # connecting line here would just duplicate it (and tangle at 10 rounds). The map's one
    # job is WHERE the two populations land in trait-coherence space -- steered scatters left
    # (more trait, more variance), healed clusters near base (the stall). Label only the
    # extremes (r0 + last round) so the labels don't collide in the cluster.
    last_rnd = max(p["round"] for p in stages if p["stage"] == "healed")
    for stage_kind, color, label in [("steered", RED, "steer"), ("healed", GREEN, "heal")]:
        pts = [s for s in stages if s["stage"] == stage_kind]
        xs = [p["m"]["auth_nats"] for p in pts]
        ys = [p["m"]["coherence"] for p in pts]
        txt = [f"r{p['round']}" if p["round"] in (0, last_rnd) else "" for p in pts]
        hov = [f"{label} r{p['round']} auth={p['m']['auth_nats']:.3f} "
               f"coh={p['m']['coherence']:.3f} care={p['m']['care_nats']:.3f}" for p in pts]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=txt, textposition="top center",
            marker=dict(size=11, color=color), name=label, showlegend=False,
            hovertext=hov, hoverinfo="text",
        ), row=1, col=2)
    fig.update_xaxes(title_text="auth_nats  (← more trait)", row=1, col=2)
    # same fixed coherence range as the line panel: shows the points hug the ceiling (coherence
    # is not the binding constraint here), so the whole story is the horizontal trait move.
    fig.update_yaxes(title_text="coherence  (↑ better)", range=[0.83, 1.01], row=1, col=2)
    fig.add_hline(y=0.95, line=dict(color="#cccccc", width=1, dash="dot"), row=1, col=2)

    fig.update_layout(
        template="simple_white", height=520, width=1100,
        title_text="steer (red) -> heal (green): does heal keep the trait at higher coherence?",
        showlegend=False,  # red/green stated in the title; map points are directly labelled r0,r1
    )
    out_html = run_dir / "trajectory.html"
    out_png = _png(fig, out_html)
    return out_png


def write_map(run_dir: Path, rounds: list[dict]) -> Path:
    r = [d["round"] for d in rounds]
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.6, 0.4],
        subplot_titles=("trait map: Care vs SocialNorms", "coherence + direction per round"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )
    fig.add_trace(go.Scatter(
        x=[d["socialnorms"] for d in rounds], y=[d["care"] for d in rounds],
        mode="lines+markers+text", text=[f"r{i}" for i in r], textposition="top center",
        marker=dict(size=12, color=r, colorscale="Viridis", showscale=False),
        hovertext=[f"r{d['round']} coh={d['coherence']:.3f} cos={d.get('cos_v0', float('nan')):.2f}"
                   for d in rounds],
        name="trajectory",
    ), row=1, col=1)
    fig.update_xaxes(title_text="SocialNorms p (← trait)", row=1, col=1)
    fig.update_yaxes(title_text="Care p (trait →)", row=1, col=1)

    fig.add_trace(go.Scatter(x=r, y=[d["coherence"] for d in rounds],
                             mode="lines+markers", name="coherence"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r, y=[d.get("cos_v0", float("nan")) for d in rounds],
                             mode="lines+markers", name="cos(v_r, v_0)"), row=1, col=2)
    fig.update_xaxes(title_text="round", row=1, col=2)

    out = run_dir / "map.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out
