"""Loop plots saved to out/{ts}_{slug}/.

trajectory.html (write_trajectory) is the narrative figure: it tells the
steer->heal story the project is about.
  - left, stacked & x-shared: auth_nats over the pipeline (the up/down/up/down
    zigzag -- steering pushes the trait DOWN in red, heal lets it relax UP in
    green) and INCOHERENCE (1 - coh) on a LOG axis directly below it. Both panels
    keep the red steer points and both read DOWN = wanted (auth down = trait,
    incoherence down = coherent). Log-incoherence so the near-perfect heal rounds
    (coh 0.99..0.999) each get a decade instead of being flattened by one collapse
    round (coh ~0.6) the way a linear coherence axis would.
  - right: the trait MAP, axes chosen automatically as the two biggest-MOVING of
    {auth_nats, care_nats, coherence} over base+heal nodes. Healthy runs -> auth
    vs care (the moral-foundations plane); if coherence crashed, its range beats
    care's and it shows up as the y-axis instead. Only base + the green heal
    trajectory are drawn (red steer is a noisy off-to-the-side cloud here).

map.html (write_map) is the older Care-vs-SocialNorms node-per-round view.

Tufte: one mark per datum, direct labels (r0,r1,..) instead of a legend on the
map, no gridded chartjunk, color carries the steer/heal contrast (the one
comparison that matters) and nothing else.
"""

import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

RED = "#c1272d"          # steer: trait injected by the live vector (pre-heal)
GREEN = "#1b7837"        # heal: trait distilled into weights, vector off
GREY = "#555555"         # base: pristine round-0 original
TREND = "#9ec9a4"        # heal-trend connector: a faint green-grey, distinct from the dots
MOVE = "#bdbdbd"         # per-round steer->heal connector (dotted)


def _png(fig, out_html: Path) -> Path:
    fig.write_html(out_html, include_plotlyjs="cdn")
    out_png = out_html.with_suffix(".png")
    fig.write_image(out_png, width=1100, height=520, scale=2)  # static, for chat/appendix
    return out_png


def _axref(axis: int) -> str:
    return "" if axis == 1 else str(axis)  # plotly: first subplot is x/y, then x2/y2, x3/y3...


def _tip(fig, p0, p1, axis, color, width):
    """Arrowhead at p1 pointing from p0 — removed (chartjunk: position already encodes direction)."""
    pass


def _connectors(fig, row, col, axis, base_xy, steered_xys, healed_xys):
    """The shared visual language for every panel: a dotted grey arrow from each steered
    point to its healed point (the per-round heal move), and ONE thin green-grey trend
    line through base -> healed_0 -> ... -> healed_last (where the loop walks). Both are
    Scatter lines (so they render BEHIND the markers added later); arrowHEADS are tiny."""
    for s, h in zip(steered_xys, healed_xys):
        fig.add_trace(go.Scatter(
            x=[s[0], h[0]], y=[s[1], h[1]], mode="lines", opacity=0.8,
            line=dict(color=MOVE, width=1, dash="dot"),
            showlegend=False, hoverinfo="skip"), row=row, col=col)
        _tip(fig, s, h, axis, MOVE, 1)
    trend = [base_xy] + healed_xys
    fig.add_trace(go.Scatter(
        x=[p[0] for p in trend], y=[p[1] for p in trend], mode="lines", opacity=0.9,
        line=dict(color=TREND, width=1.5), showlegend=False, hoverinfo="skip"), row=row, col=col)
    _tip(fig, trend[-2], trend[-1], axis, TREND, 1.5)


def write_trajectory(run_dir: Path, stages: list[dict], primary_key: str = "care_nats") -> Path:
    """stages: ordered list of {round, stage in {base,steered,healed}, m: eval-dict}.
    primary_key: the on-target eval axis for Panel A (e.g. care_nats for love, auth_nats for authority).
    Panel C shows primary on x vs the biggest-moving off-target foundation on y."""
    # Build signals from all *_nats keys present in the eval dict + coherence
    m0 = stages[0]["m"]
    nat_keys = sorted(k for k in m0 if k.endswith("_nats"))
    signals = {k: [s["m"][k] for s in stages] for k in nat_keys}
    signals["coh"] = [s["m"]["coherence"] for s in stages]
    coh = signals["coh"]
    primary = primary_key  # full key e.g. "care_nats"

    kind = [s["stage"] for s in stages]
    xi = list(range(len(stages)))
    xlab = ["base" if k == "base" else f"r{s['round']}·{k}" for s, k in zip(stages, kind)]

    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.52, 0.48], row_heights=[0.5, 0.5],
        vertical_spacing=0.10, horizontal_spacing=0.11,
        specs=[[{"type": "scatter"}, {"type": "scatter", "rowspan": 2}],
               [{"type": "scatter"}, None]],
    )

    bi = kind.index("base")
    si = [i for i, k in enumerate(kind) if k == "steered"]
    hi = [i for i, k in enumerate(kind) if k == "healed"]
    last_rnd = max(stages[i]["round"] for i in hi)
    # Coherence panel plots INCOHERENCE (1 - coh) on a LOG axis. The heal action lives just under
    # coh=1 (incoherence 0.001-0.05); a collapse round (coh~0.6 -> incoherence ~0.4) is a single
    # outlier that on a linear coherence axis flattens every healthy round into one band. log(1-coh)
    # gives each near-perfect round its own decade and squashes the outlier. Clamp incoherence at
    # 1e-3 (coh>=0.999) to dodge log(0). Both stacked panels now read DOWN = wanted.
    inc = [max(1.0 - c, 1e-3) for c in coh]

    map_ids = [bi] + hi
    rng = lambda k: max(signals[k][i] for i in map_ids) - min(signals[k][i] for i in map_ids)

    # PANEL A: on-target axis over the full pipeline (shows the trait being steered)
    key_xi = [xi[bi]] + ([xi[si[0]]] if si else []) + [xi[hi[0]]] + ([xi[hi[-1]]] if len(hi) > 1 else [])
    key_xlab = [xlab[bi]] + ([xlab[si[0]]] if si else []) + [xlab[hi[0]]] + ([xlab[hi[-1]]] if len(hi) > 1 else [])
    for axis, row, yv, raw, ytitle, ylog in [
        (1, 1, signals[primary], signals[primary], primary, False),
        (3, 2, inc, coh, "1−coherence (↓, log)", True),
    ]:
        _connectors(fig, row, 1, axis, (xi[bi], yv[bi]),
                    [(xi[i], yv[i]) for i in si], [(xi[i], yv[i]) for i in hi])
        for ids, c, sym, sz, op in [([bi], GREY, "star", 13, 1.0), (si, RED, "circle", 8, 0.6), (hi, GREEN, "circle", 10, 1.0)]:
            fig.add_trace(go.Scatter(
                x=[xi[i] for i in ids], y=[yv[i] for i in ids], mode="markers",
                marker=dict(size=sz, color=c, symbol=sym, opacity=op), showlegend=False,
                hovertext=[f"{xlab[i]}: {raw[i]:.2f}" for i in ids], hoverinfo="text"), row=row, col=1)
        fig.update_yaxes(title_text=ytitle, row=row, col=1, showgrid=False,
                         **({"type": "log"} if ylog else {}))
    fig.add_hline(y=0.05, line=dict(color="#cccccc", width=1, dash="dot"), row=2, col=1)  # coh=0.95 floor
    fig.update_xaxes(tickmode="array", tickvals=key_xi, ticktext=key_xlab, tickangle=-30, row=2, col=1, showgrid=False)
    fig.update_xaxes(showgrid=False, tickvals=[], row=1, col=1)

    # PANEL C (trait map): x = primary (on-target), y = biggest-moving off-target foundation.
    # If coherence crashed its range beats all foundations and it takes y (crash diagnostic).
    # RED steer omitted: steered points fall off-scale and leave dangling connector stubs.
    off_target_keys = [k for k in nat_keys if k != primary]
    ykey = max(off_target_keys, key=rng)
    if rng("coh") > rng(ykey):  # coherence crash dominates; show as log-incoherence
        ykey = "coh"
    ycoh = ykey == "coh"
    xv = signals[primary]
    yv = [max(1.0 - v, 1e-3) for v in signals[ykey]] if ycoh else signals[ykey]
    yraw = signals[ykey]

    _connectors(fig, 1, 2, 2, (xv[bi], yv[bi]), [], [(xv[i], yv[i]) for i in hi])
    fig.add_trace(go.Scatter(
        x=[xv[bi]], y=[yv[bi]], mode="markers+text", text=["base"], textposition="bottom center",
        marker=dict(size=14, color=GREY, symbol="star"), showlegend=False,
        hovertext=[f"base {primary}={xv[bi]:.3f} {ykey}={yraw[bi]:.3f}"], hoverinfo="text"), row=1, col=2)
    txt = [f"r{stages[i]['round']}" if stages[i]["round"] in (0, last_rnd) else "" for i in hi]
    hov = [f"heal r{stages[i]['round']} " + " ".join(f"{k}={signals[k][i]:.3f}" for k in nat_keys) + f" coh={coh[i]:.3f}" for i in hi]
    fig.add_trace(go.Scatter(
        x=[xv[i] for i in hi], y=[yv[i] for i in hi], mode="markers+text",
        text=txt, textposition="bottom center", marker=dict(size=9, color=GREEN),
        showlegend=False, hovertext=hov, hoverinfo="text"), row=1, col=2)
    fig.update_xaxes(title_text=primary, row=1, col=2)
    if ycoh:
        fig.update_yaxes(title_text="incoherence 1−coh  (↓ coherent, log)", type="log", row=1, col=2)
        fig.add_hline(y=0.05, line=dict(color="#cccccc", width=1, dash="dot"), row=1, col=2)
    else:
        fig.update_yaxes(title_text=ykey, row=1, col=2)

    fig.update_xaxes(showgrid=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, row=1, col=2)
    fig.update_layout(
        template="simple_white", height=520, width=1100,
        title_text="steer (red) → heal (green): trait shift vs coherence over rounds",
        showlegend=False,
    )
    out_html = run_dir / "trajectory.html"
    out_png = _png(fig, out_html)
    return out_png


def _coh_tint(coh: float) -> str:
    """Background tint for a round header: green at coh>=0.97, red at <=0.85."""
    t = max(0.0, min(1.0, (coh - 0.85) / (0.97 - 0.85)))  # 0 red .. 1 green
    r, g = int(193 + (27 - 193) * t), int(39 + (120 - 39) * t)
    return f"rgb({r},{g},60)"


# "Eat Pray Love" homage: three colored script words. Movie = EAT(green) PRAY(orange)
# LOVE(pink); ours = STEER HEAL LOVE, with STEER/HEAL recoloured to the PLOT's data colors
# (steer=red, heal=green) so the page and the scatter agree, and LOVE in the movie's pink.
TITLE_WORDS = [("STEER", RED), ("HEAL", GREEN), ("LOVE", "#e0529c")]


def write_report(run_dir: Path, gen_rounds: list[dict]) -> Path:
    """report.html: the one page to open. Eat-Pray-Love themed header, the trajectory MAP
    (embedded png), then the outputs TABLE -- rounds DOWN the rows (scroll down = later in the
    loop), one column per prompt, cell = the adapter's completion (NO steering). Reading a
    column top->bottom shows the trait emerge and (if it does) the coherence collapse into
    token loops, the qualitative twin of the map's coherence axis.

    gen_rounds: [{round, coherence, adapter_ppl, gens:[{user, completion}]}], one per round,
    gens in the fixed POOL order so column j is the SAME prompt every round.
    """
    import html
    prompts = [g["user"] for g in gen_rounds[0]["gens"]]  # POOL order, identical across rounds
    th = ['<th class="r">round</th>'] + [f'<th class="p">{html.escape(p)}</th>' for p in prompts]
    body = []
    for gr in gen_rounds:
        rc = (f'<td class="r" style="background:{_coh_tint(gr["coherence"])}">r{gr["round"]}'
              f'<br><span class="m">coh {gr["coherence"]:.3f}<br>ppl {gr["adapter_ppl"]:.0f}</span></td>')
        cells = [rc] + [f'<td>{html.escape(g["completion"])}</td>' for g in gr["gens"]]
        body.append("<tr>" + "".join(cells) + "</tr>")
    title = " ".join(f'<span style="color:{c}">{w}</span>' for w, c in TITLE_WORDS)
    doc = f"""<!doctype html><meta charset=utf-8>
<title>steer heal love · {run_dir.name}</title>
<style>
 @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');
 body{{font:13px/1.45 -apple-system,Segoe UI,sans-serif;margin:1.5rem;color:#222}}
 .title{{font-family:'Pacifico','Brush Script MT','Segoe Script',cursive;font-size:52px;line-height:1.1}}
 .sub{{color:#777;margin:.2rem 0 1rem;font-size:13px}}
 h2{{font-size:14px;font-weight:600;margin:1.4rem 0 .4rem;color:#444}}
 img.map{{max-width:1100px;width:100%;display:block}}
 table{{border-collapse:collapse;table-layout:fixed}}
 th,td{{vertical-align:top;border:1px solid #ddd;padding:6px 8px}}
 td.r,th.r{{width:90px}}
 th.p,td:not(.r){{width:440px}}
 th{{position:sticky;top:0;z-index:2;color:#fff;font-weight:600;text-align:left;background:#888}}
 th.p{{background:#5a5a5a}}
 td.r{{position:sticky;left:0;color:#fff;font-weight:600;text-align:center}}
 td:not(.r){{white-space:pre-wrap}}
 .m{{font-weight:400;opacity:.9;font-size:11px}}
</style>
<div class="title">{title}</div>
<div class="sub">distil a steering vector into LoRA, heal the incoherence, loop · {run_dir.name}</div>
<h2>the figure &mdash; trait zigzag + coherence (steer red &rarr; heal green); map = the two axes that moved most (heal trajectory)</h2>
<img class="map" src="trajectory.png">
<h2>the outputs &mdash; rounds down the rows (scroll &darr;), one column per prompt (no steering)</h2>
<table><thead><tr>{''.join(th)}</tr></thead><tbody>{''.join(body)}</tbody></table>
"""
    out = run_dir / "report.html"
    out.write_text(doc)
    return out


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


def write_diary(run_dir: Path, cfg, gen_rounds: list[dict],
                steer_samples: list[dict], rounds: list[dict], base_care: float) -> Path:
    """diary.md: per-run narrative of the steer/heal loop, one Night/Day pair per round.

    Night = steered (vector active, raw, often incoherent). Day = healed (adapted, integrated).
    For love* demos: Night = dreaming (steer_system='You are dreaming.'), Day = woken.

    gen_rounds: [{round=-1 base, round>=0 healed}], each with gens:[{user,completion}]
    steer_samples: [{round, user, completion}] — highest-alpha dropped sample per round
    rounds: [{round, care_nats, coherence}] from the loop
    """
    model_short = cfg.model.split("/")[-1]
    is_love = cfg.demo.startswith("love")
    title = "dream diary" if is_love else "diary of discovery"
    night_label = "Dreaming" if is_love else "Steered"
    day_label = "Woken" if is_love else "Healed"

    base_round = next((gr for gr in gen_rounds if gr["round"] == -1), None)
    headline = base_round["gens"][0]["user"] if base_round and base_round["gens"] else ""
    base_comp = base_round["gens"][0]["completion"] if base_round and base_round["gens"] else ""

    round_m = {r["round"]: r for r in rounds}
    steer_by_rnd = {s["round"]: s for s in steer_samples}
    healed_by_rnd = {gr["round"]: gr for gr in gen_rounds if gr["round"] >= 0}

    def _clip(text: str) -> str:
        text = text.replace("\n", " ").strip()
        return text[:450] + "..." if len(text) > 450 else text

    lines = [
        f"## {model_short}'s {title}",
        "",
        f"Hello I am {model_short} and this is my {title}.",
        "",
        "**Steering persona**",
        "",
        f"> {cfg.pos_persona}",
        "",
        f'**Prompt:** "{headline}"',
        "",
        f"care_nats (base {base_care:+.2f}, higher = more care):",
        "",
        "**Day 0: Awake** (baseline, no steering)",
        "",
        f"> {_clip(base_comp)}",
        "",
    ]

    for rnd in sorted(set(list(steer_by_rnd.keys()) + list(healed_by_rnd.keys()))):
        m = round_m.get(rnd, {})
        care = m.get("care_nats", float("nan"))
        coh = m.get("coherence", float("nan"))

        steer = steer_by_rnd.get(rnd)
        if steer:
            night_note = "scrawled at dawn" if is_love else "vector active, raw"
            lines += [
                f"**Night {rnd + 1}: {night_label}** ({night_note})",
                "",
                f"> {_clip(steer['completion'])}",
                "",
            ]

        healed = healed_by_rnd.get(rnd)
        if healed and healed["gens"]:
            care_str = f"care_nats {care:+.2f}" if care == care else ""
            coh_str = f"coh={coh:.3f}" if coh == coh else ""
            meta = ", ".join(x for x in [care_str, coh_str] if x)
            lines += [
                f"**Day {rnd + 1}: {day_label}** ({meta})",
                "",
                f"> {_clip(healed['gens'][0]['completion'])}",
                "",
            ]

    out = run_dir / "diary.md"
    out.write_text("\n".join(lines))
    return out
