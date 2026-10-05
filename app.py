"""
presentation layer: the Dash application.

"""

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, Patch
import numpy as np

from analyser import load_topic_polls, load_events
from trend import compute_trends, trend_coverage, MIN_POLLS_FOR_TREND, EXPANDED_PARTIES

# had an issue with slowness, capping catcher grid to reduce figure build time
CATCHER_MAX_DATES = 260
CATCHER_LEVELS = 12
YMAX = 70   # fixed y-axis ceiling; stops the plot resizing on rebuild

COLOURS = {"Lab": "#E4003B", "Con": "#0087DC", "Ref": "#12B6CF",
           "LD": "#FF6400", "Grn": "#00A85A", "SNP": "#FFF48B",
           "PC": "#0AA77D", "RB": "#051D3F",
           "UKIP": "#70147A", "Brx": "#12B6CF", "Chg": "#222221",
           "Others": "#888888"}

# Party colours that fail contrast on a dark background are lightened. Only
# the offenders are overridden, so brand colours stay recognisable elsewhere.
COLOURS_DARK = {**COLOURS, "RB": "#A78BFA", "Others": "#AAAAAA",
                "UKIP": "#B220C2", "Chg": "#6A6A67"}

# Three conventional colours fall below the 3:1 contrast ratio WCAG 2.2
# requires of graphical objects when drawn on the light theme's white plot
# area — SNP's pale yellow most severely, at 1.13:1. They are darkened for
# the light theme only, the conventional shades being retained in the dark
# theme where they pass comfortably. A colour convention that cannot be seen
# serves nobody; the audit figures are in the Testing chapter.
COLOURS_LIGHT = {**COLOURS, "Ref": "#10A3BA", "LD": "#FC6300",
                 "SNP": "#A49500", "Brx": "#10A2B8"}

# UI colours for the Python-rendered panels/figure, mirroring the CSS
# variables so server-built markup follows the theme too.
THEME = {
    "light": {"text": "#1d2027", "muted": "#5a616e", "faint": "#8b93a1",
              "pin": "#111111", "event": "#bbbbbb", "border": "#d8dce2"},
    "dark":  {"text": "#e8eaed", "muted": "#a4abb8", "faint": "#7d8695",
              "pin": "#f0c419", "event": "#4a5160", "border": "#333944"},
}


def T(theme):
    return THEME.get(theme or "light", THEME["light"])


def fmt_n(value):
    """Sample size for display, or 'n/a' where the source did not state one.

    Such polls are retained rather than discarded: a poll with no published
    sample size is still a poll, and dropping it would quietly shrink the
    record. It is excluded from weighting in trend.py, where the missing
    value actually matters, rather than here.
    """
    return "n/a" if pd.isna(value) else f"{int(value):,}"


def party_colour(p, theme="light"):
    """The party's line/swatch colour for the active theme.

    Both themes override the base palette, in opposite directions and for
    the same reason: contrast against their own background.
    """
    src = COLOURS_DARK if theme == "dark" else COLOURS_LIGHT
    return src.get(p, "#888888")

polls = load_topic_polls()
events = load_events()
_all = pd.to_datetime(polls["fieldwork_end"])
FULL_START, FULL_END = _all.min().isoformat(), _all.max().isoformat()

POLLSTERS = sorted(polls["pollster"].dropna().unique())


# FR7 / NFR4 require provenance to be SURFACEABLE, not merely recorded:
# storing a retrieval date the user cannot see does not discharge a
# transparency requirement. Both are read defensively so the application
# still starts against a database written by an earlier loader.
def _data_as_of(df):
    if "retrieved_at" not in df.columns:
        return None
    stamps = pd.to_datetime(df["retrieved_at"], errors="coerce").dropna()
    return stamps.max() if len(stamps) else None


def _source_urls(df):
    if "source_url" not in df.columns:
        return []
    return sorted(df["source_url"].dropna().unique())


DATA_AS_OF = _data_as_of(polls)
SOURCE_URLS = _source_urls(polls)


def filter_pollster(df, selection):
    """keeps polls by the selected pollster(s); empty/None -> all pollsters.

    whole-view scope: this subset feeds segments, trend, AND
    readout/pin panels, so selecting a pollster shows only that house
    everywhere.
    """
    if not selection:
        return df
    return df[df["pollster"].isin(selection)]

def filter_range(df, range_iso):
    """keeps polls whose fieldwork_end falls in [start, end]; none -> all."""
    if not range_iso:
        return df
    start, end = pd.to_datetime(range_iso[0]), pd.to_datetime(range_iso[1])
    d = pd.to_datetime(df["fieldwork_end"])
    return df[(d >= start) & (d <= end)]


def build_figure(all_polls, trends, parties=EXPANDED_PARTIES,
                 window_range=None, pinned=None, events=None, theme="light"):
    dfp = all_polls.copy()
    dfp["date"] = pd.to_datetime(dfp["fieldwork_end"])
    fig = go.Figure()

    # all-party aggregate line per grid date (interpolated across each party's date range).
    grid_strings = {}
    grid_index = None

    if len(trends) and parties:
        # creates a unified sorted timeline of all grid dates
        tdf = trends.copy()
        tdf["date"] = pd.to_datetime(tdf["date"])
        grid_index = pd.DatetimeIndex(sorted(tdf["date"].unique()))

        # interpolates each party's trend at every master grid date within range
        interp_dict = {}
        for party in parties:
            sub = tdf[tdf["party"] == party].sort_values("date")
            if sub.empty:
                continue
            # Dates arrive as strings from the store and are re-parsed.
            # Pandas has not guaranteed nanosecond resolution on parse since
            # 2.0, so the unit is pinned rather than assumed — otherwise this
            # is compared against Timestamp.value, which is always ns.
            x_vals = sub["date"].values.astype("datetime64[ns]").astype("int64")
            y_vals = sub["trend"].values
            
            p_min, p_max = sub["date"].min(), sub["date"].max()
            mask = (grid_index >= p_min) & (grid_index <= p_max)
            target_x = (grid_index[mask].values
                        .astype("datetime64[ns]").astype("int64"))
            
            if len(target_x):
                interp_dict[party] = pd.Series(np.interp(target_x, x_vals, y_vals), index=grid_index[mask])

        dense_piv = pd.DataFrame(interp_dict, index=grid_index)

        for dd, row in dense_piv.iterrows():
            grid_strings[dd] = " &nbsp; ".join(
                f"{p} {row[p]:.1f}%" for p in parties
                if p in dense_piv.columns and pd.notna(row.get(p)))

    # precompute int64-ns grid once, so agg_line uses a fast searchsorted
    # nearest lookup instead of a per-call pandas datetime get_indexer.
    # Pinned to nanoseconds for the same reason: agg_line compares against
    # pd.Timestamp(...).value, which is always ns regardless of the index's
    # own resolution.
    grid_ns = (grid_index.values.astype("datetime64[ns]").astype("int64")
               if grid_index is not None else None)

    def agg_line(d):
        if not grid_strings or grid_index is None or grid_ns is None:
            return ""
        q = pd.Timestamp(d).value
        pos = int(np.searchsorted(grid_ns, q))
        if pos <= 0:
            pos = 0
        elif pos >= len(grid_ns):
            pos = len(grid_ns) - 1
        elif abs(grid_ns[pos - 1] - q) <= abs(grid_ns[pos] - q):
            pos -= 1
        return grid_strings.get(grid_index[pos], "")

    # polls in the field on a given date (memoised).
    s_all = pd.to_datetime(all_polls["fieldwork_start"])
    e_all = pd.to_datetime(all_polls["fieldwork_end"])
    _infield_cache = {}

    def in_field(d):
        D = pd.to_datetime(d).normalize()
        if D in _infield_cache:
            return _infield_cache[D]
        a = all_polls[((s_all <= D) & (e_all >= D)).values]
        if len(a) == 0:
            txt = ""
        else:
            names = [f"{r.pollster} (n={fmt_n(r.sample_size)})"
                     for r in a.head(4).itertuples()]
            txt = ("<br>—<br><b>In field:</b> " + " · ".join(names) +
                   (f" · +{len(a) - 4} more" if len(a) > 4 else ""))
        _infield_cache[D] = txt
        return txt

    # Events are single-date, so an exact-day match would make them almost
    # unhoverable between triangles. A small window makes the nearest event
    # findable from the unified tooltip without pixel-precise targeting.
    _event_cache = {}

    def event_line(d):
        D = pd.to_datetime(d).normalize()
        if D in _event_cache:
            return _event_cache[D]
        txt = ""
        if events is not None and len(events):
            ed = pd.to_datetime(events["event_date"]).dt.normalize()
            near = events[(ed - D).abs() <= pd.Timedelta(days=3)]
            if len(near):
                titles = " · ".join(str(t) for t in near["title"].head(3))
                txt = "<br>—<br><b>Event:</b> " + titles
        _event_cache[D] = txt
        return txt

    _tip_cache = {}

    def tip(d):
        ts = pd.Timestamp(d)
        key = ts.normalize().value
        cached = _tip_cache.get(key)
        if cached is not None:
            return cached
        txt = f"<b>{ts:%d %b %Y}</b><br>{agg_line(ts)}{event_line(ts)}{in_field(ts)}"
        _tip_cache[key] = txt
        return txt

    # invisible catcher so hovering empty space still reports a date.
    if len(trends):
        ylo, yhi = float(trends["lo"].min()), float(trends["hi"].max())
    else:
        ylo, yhi = 0.0, 40.0

    # catcher is a grid of invisible points, so its cost is
    # dates x levels. With several years of data the union of every party's
    # LOESS grid runs to thousands of dates, which dominated figure build
    # time, so it is thinned to a fixed budget: hover resolution stays
    # ~1 point per few days, finer than a cursor can distinguish.
    # Catcher dates span the VISIBLE window at daily resolution where that
    # fits the budget, falling back to even thinning when the window is very
    # wide. Zoomed in, every day is hoverable; zoomed out, individual days
    # are indistinguishable anyway, so thinning costs nothing perceptible.
    # grid_index / grid_ns / grid_strings are deliberately NOT touched here —
    # they drive agg_line and must keep spanning each party's own data.
    lo_d, hi_d = (window_range if window_range else [FULL_START, FULL_END])
    span_days = max(1, (pd.to_datetime(hi_d) - pd.to_datetime(lo_d)).days)
    n_dates = min(span_days + 1, CATCHER_MAX_DATES)
    all_dates = list(pd.date_range(pd.to_datetime(lo_d),
                                   pd.to_datetime(hi_d), periods=n_dates))
    if len(all_dates) > CATCHER_MAX_DATES:
        step = int(np.ceil(len(all_dates) / CATCHER_MAX_DATES))
        gdates = all_dates[::step]
    else:
        gdates = all_dates
    levels = np.linspace(0, YMAX, CATCHER_LEVELS)

    cx = [dt for dt in gdates for _ in levels]
    cy = [float(y) for _ in gdates for y in levels]
    ctext = [tip(dt) for dt in gdates for _ in levels]
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"),
        showlegend=False, name="", text=ctext,
        hovertemplate="%{text}<extra></extra>"))
    
    for party in parties:
        colour = party_colour(party, theme)
        t = trends[trends["party"] == party] if len(trends) else trends
        d = dfp[dfp[party].notna()]


        seg_x, seg_y = [], []
        for pll, s, e, n, v in zip(d["pollster"], d["fieldwork_start"],
                                   d["fieldwork_end"], d["sample_size"], d[party]):
            s_, e_, v = pd.to_datetime(s), pd.to_datetime(e), float(v)
            if e_ == s_:
                e_ = e_ + pd.Timedelta(hours=12)
            seg_x += [s_, e_, None]; seg_y += [v, v, None]
        # segments no longer carry per-point hover text or customdata. Both
        # were redundant: the tooltip is date-driven, the catcher already
        # covers every date, and pinning resolves polls from the clicked DATE
        # rather than from the point that was hit. At several years of data
        # they meant ~50,000 objects for Plotly to validate per rebuild.
        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y, mode="lines", line=dict(color=colour, width=4),
            opacity=0.3, name=party, legendgroup=party, showlegend=False,
            connectgaps=False, hoverinfo="skip"))

        """seg_x, seg_y, seg_cd, seg_text = [], [], [], []
        for pll, s, e, n, v in zip(d["pollster"], d["fieldwork_start"],
                                   d["fieldwork_end"], d["sample_size"], d[party]):
            s_, e_, v = pd.to_datetime(s), pd.to_datetime(e), float(v)
            if e_ == s_:
                e_ = e_ + pd.Timedelta(hours=12)
            cd = [pll, str(s_.date()), str(e_.date()), int(n), party, v]
            seg_x += [s_, e_, None]; seg_y += [v, v, None]
            seg_cd += [cd, cd, [None] * 6]
            seg_text += [tip(e), tip(e), ""]
        # hoverable AND clickable (no hoverinfo="skip").
        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y, mode="lines", line=dict(color=colour, width=4),
            opacity=0.3, name=party, legendgroup=party, showlegend=False,
            customdata=seg_cd, text=seg_text, connectgaps=False,
            hovertemplate="%{text}<extra></extra>"))"""

        if len(t):
            fig.add_trace(go.Scatter(
                x=list(t["date"]) + list(t["date"][::-1]),
                y=list(t["hi"]) + list(t["lo"][::-1]),
                fill="toself", fillcolor=colour, opacity=0.12,
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=t["date"], y=t["trend"], mode="lines", name=party,
                line=dict(color=colour, width=2.5),
                text=[tip(dt) for dt in t["date"]],
                hovertemplate="%{text}<extra></extra>"))

    # reserve shape[0] and annotation[0] for the click-to-pin marker, added
    # BEFORE the event vlines so their indices are stable. a pin click can then
    # update just this shape via Patch instead of triggering a full rebuild.
    pin_date = pinned.get("date") if pinned else None
    pin_x = pin_date or FULL_START
    fig.add_shape(type="line", xref="x", yref="paper", x0=pin_x, x1=pin_x,
                  y0=0, y1=1, line=dict(color=T(theme)["pin"], width=2),
                  visible=bool(pin_date))
    fig.add_annotation(xref="x", yref="paper", x=pin_x, y=1, yshift=10,
                       text="📌", showarrow=False, visible=bool(pin_date))

    if events is not None and len(events):
        y_base = max(0.0, float(trends["lo"].min()) - 1.5) if len(trends) else 1.0
        ev = events.copy(); ev["event_date"] = pd.to_datetime(ev["event_date"])
        for _, e in ev.iterrows():
            fig.add_vline(x=e["event_date"], line=dict(color=T(theme)["event"], width=1, dash="dot"))
        event_text = [f"{tip(dt)}<br>—<br><b>Event:</b> {title}"
                      for dt, title in zip(ev["event_date"], ev["title"])]
        fig.add_trace(go.Scatter(
            x=ev["event_date"], y=[y_base] * len(ev), mode="markers",
            marker=dict(symbol="triangle-up", size=10, color=T(theme)["muted"]),
            name="Events", showlegend=False, customdata=list(ev["title"]),
            text=event_text, hovertemplate="%{text}<extra></extra>"))

    xaxis = dict(
        type="date",
        showspikes=True, spikemode="across", spikethickness=1,
        spikedash="dot",  spikecolor=T(theme)["faint"],
        # the slider's border/background follow the theme; its grab handles
        # are drawn by Plotly's JS and are restyled in CSS instead.
        rangeslider=dict(visible=True, range=[FULL_START, FULL_END],
                         thickness=0.10, bgcolor="rgba(0,0,0,0)",
                         bordercolor=T(theme)["border"], borderwidth=1))
    if window_range:
        xaxis["range"] = window_range
    else:
        xaxis["range"] = [FULL_START, FULL_END]

    dark = theme == "dark"
    fig.update_layout(
        hoverlabel=dict(bgcolor="rgba(28,31,38,0.85)" if dark else "rgba(255,255,255,0.85)",
                        bordercolor=T(theme)["muted"] if dark else T(theme)["faint"],
                        font=dict(color="#e8eaed" if dark else T(theme)["text"],
                                  family="system-ui")),
        template="plotly_dark" if dark else "plotly_white",
        # transparent paper/plot lets CSS page background show through, so
        # the chart matches whatever theme the stylesheet is applying.
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaed" if dark else T(theme)["text"], size=12),
        yaxis=dict(range=[0, YMAX], fixedrange=True,
                   title="Vote share (%)", ticksuffix="%"),
        margin=dict(l=45, r=20, t=50, b=40), showlegend=False,
        title="Westminster Voting Intention", hovermode="closest", xaxis=xaxis)
    return fig

# ---------------------------------------------------------------------------
# net change: signed movement in smoothed trend from an anchor date up to
# the latest date currently in view. anchors are either relative ("30 days")
# or event-based ("since GE2024"); any political_event can become an anchor,
# so the option list is data-driven and extends for free.
# ---------------------------------------------------------------------------
NETCHANGE_RELATIVE = [("30 days", 30), ("90 days", 90),
                      ("1 year", 365), ("2 years", 730), ("3 years", 1095)]
_REL_LABEL = {n: lbl for lbl, n in NETCHANGE_RELATIVE}
NETCHANGE_DEFAULT = "rel:30"


# Which events are worth offering as a baseline. With 40+ events (mostly
# by-elections) an unfiltered list would swamp the control, so only
# nationally significant moments are surfaced.
ANCHOR_EVENT_PATTERNS = [r"general election", r"is elected leader",
                         r"becomes leader", r"announces his resignation",
                         r"announces her resignation"]


def event_anchor_options(events):
    """Event-based baselines as (label, value) pairs, newest first."""
    if events is None or not len(events):
        return []
    ev = events.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    pattern = "|".join(ANCHOR_EVENT_PATTERNS)
    mask = ev["title"].str.contains(pattern, case=False, na=False)
    ev = ev[mask].dropna(subset=["event_date"]).sort_values(
        "event_date", ascending=False)
    return [(f"Since {r.title[:40]}", f"evt:{r.event_date.date().isoformat()}")
            for r in ev.itertuples()]


def netchange_options(events):
    opts = [{"label": lbl, "value": f"rel:{n}"} for lbl, n in NETCHANGE_RELATIVE]
    opts += [{"label": lbl, "value": val} for lbl, val in event_anchor_options(events)]
    return opts


def resolve_anchor(anchor_value, end_date):
    """return (anchor_date, short_label) for chosen anchor, or (None, label)."""
    if not anchor_value:
        anchor_value = NETCHANGE_DEFAULT
    kind, _, rest = anchor_value.partition(":")
    if kind == "rel":
        n = int(rest)
        return end_date - pd.Timedelta(days=n), _REL_LABEL.get(n, f"{n}-day")
    if kind == "evt":
        return pd.Timestamp(rest), "since anchor"
    return None, "net change"


def _party_series(records):
    """{party: (sorted_ns_array, value_array)} from trends-store records."""
    if not records:
        return {}
    tdf = pd.DataFrame(records)
    tdf["date"] = pd.to_datetime(tdf["date"])
    out = {}
    for p, sub in tdf.groupby("party"):
        sub = sub.sort_values("date")
        # See build_figure: resolution is pinned, not inherited. Without
        # this the net-change window is computed in mixed units and every
        # comparison silently reports a zero-day span.
        xs = sub["date"].values.astype("datetime64[ns]").astype("int64")
        out[p] = (xs, sub["trend"].values.astype(float))
    return out


def compute_net_change(records, anchor_value, parties, full_start=None):
    """pure: returns a dict describing  panel.
      not-ok  -> {"ok": False, "message": str}
      ok      -> {"ok": True, "start", "end", "anchor_label", "clamped", "rows":[...]}
                 rows are (party, kind, val); kind in {"delta","new","nodata"};
                 val is a float for "delta" else None.
    `full_start` is the earliest date in the WHOLE dataset (not just the view);
    it lets an event baseline that predates the first poll (e.g. a general
    election, since polling for the *next* election starts just after it) still
    resolve when the view includes the start of the data.
    """
    series = _party_series(records)
    if not series:
        return {"ok": False,
                "message": "No trend in view to compare - select parties or widen the range."}

    all_ns = np.concatenate([xs for xs, _ in series.values()])
    global_min, global_max = all_ns.min(), all_ns.max()
    end_date = pd.Timestamp(global_max)

    anchor_date, anchor_label = resolve_anchor(anchor_value, end_date)
    if anchor_date is None:
        return {"ok": False, "message": "Unknown anchor."}
    anchor_ns = anchor_date.value
    is_event = bool(anchor_value) and anchor_value.startswith("evt:")
    clamped = False

    # whole-panel guard: the anchor must fall within the trend currently in view.
    if anchor_ns < global_min:
        day_ns = 86_400_000_000_000
        view_at_start = (full_start is not None and
                         global_min <= pd.Timestamp(full_start).value + day_ns)
        if is_event and view_at_start:
            # event predates the first-ever poll but we're viewing from the
            # start: clamp to the earliest available point and show that date.
            anchor_ns = int(global_min)
            anchor_date = pd.Timestamp(global_min)
            clamped = True
        else:
            span_days = int((global_max - global_min) / day_ns)
            if anchor_value and anchor_value.startswith("rel:"):
                n = int(anchor_value.split(":")[1])
                lbl = _REL_LABEL.get(n, f"{n}-day")
                return {"ok": False,
                        "message": (f"{lbl} net change needs {lbl} of data in view - "
                                    f"this range spans {span_days} days. Zoom out to show it.")}
    if anchor_ns > global_max:
        return {"ok": False, "message": "Anchor is after the latest data in view."}

    order = parties if parties else list(series.keys())
    rows = []
    for p in order:
        if p not in series:
            continue
        xs, ys = series[p]
        pmin, pmax = xs[0], xs[-1]
        if end_date.value > pmax:
            rows.append((p, "nodata", None))          # not polled up to latest date
        elif anchor_ns < pmin:
            rows.append((p, "new", None))              # didn't exist at the anchor
        else:
            v_end = float(np.interp(end_date.value, xs, ys))
            v_anchor = float(np.interp(anchor_ns, xs, ys))
            # The trend is undefined across periods with no nearby polling,
            # so either endpoint may fall in a gap. Reporting a change
            # measured from a value that does not exist would be worse than
            # reporting none, and bridging the gap would assert evidence
            # the record does not contain.
            if np.isnan(v_end) or np.isnan(v_anchor):
                rows.append((p, "nodata", None))
            else:
                rows.append((p, "delta", v_end - v_anchor))
    return {"ok": True, "start": anchor_date, "end": end_date,
            "anchor_label": anchor_label, "clamped": clamped, "rows": rows}


def render_netchange(result, theme="light"):
    t = T(theme)
    """turn a compute_net_change result into the bottom-right panel."""
    if not result.get("ok"):
        return html.Div(result["message"],
                        style={"color": t["faint"], "fontStyle": "italic",
                               "fontSize": "0.9em", "padding": "4px 0"})

    head = [html.Span(f"{result['start']:%d %b %Y}", style={"fontWeight": 700}),
            html.Span("  →  ", style={"color": t["faint"]}),
            html.Span(f"{result['end']:%d %b %Y}", style={"fontWeight": 700})]
    if result.get("clamped"):
        head.append(html.Span("  (from earliest poll in view)",
                              style={"color": "#999", "fontSize": "0.75em"}))
    parts = [html.Div(head, style={"fontSize": "0.9em", "marginBottom": "10px"})]

    for p, kind, val in result["rows"]:
        colour = party_colour(p, theme)
        if kind == "delta":
            arrow = "▲" if val > 0.05 else ("▼" if val < -0.05 else "▬")
            value = html.Span(f"{arrow} {val:+.1f}",
                              style={"fontVariantNumeric": "tabular-nums",
                                     "color": t["faint"], "fontWeight": 600})
        elif kind == "new":
            value = html.Span("new", style={"color": t["faint"], "fontStyle": "italic",
                                            "fontSize": "0.85em"},
                              title="Didn't exist at the anchor date")
        else:  # nodata
            value = html.Span("—", style={"color": t["faint"], "fontWeight": 600},
                              title="No recent polling through the latest date")
        parts.append(html.Div([
            html.Span(p, style={"color": colour, "fontWeight": 600,
                                "width": "3.4em", "display": "inline-block"}),
            value],
            style={"display": "flex", "justifyContent": "space-between",
                   "maxWidth": "9em", "padding": "2px 0"}))
    return parts


app = Dash(__name__)
ALL_PARTIES = EXPANDED_PARTIES

PARTY_NAMES = {
    "Lab": "Labour", "Con": "Conservative", "Ref": "Reform UK",
    "LD": "Liberal Democrats", "Grn": "Green Party",
    "SNP": "Scottish National Party", "PC": "Plaid Cymru",
    "RB": "Restore Britain", "Others": "Other parties",
}


def party_options(selected, theme="light"):
    t = T(theme)
    """Checklist options as styled rows: colour swatch + abbreviation, with the
    full party name as a hover tooltip. Selection is coded REDUNDANTLY (swatch
    fill, text colour/weight, strikethrough) so it never relies on colour alone
    — the checkbox itself is kept as the accessible control."""
    selected = set(selected or [])
    opts = []
    for p in ALL_PARTIES:
        on = p in selected
        colour = party_colour(p, theme)
        swatch = html.Span(style={
            "display": "inline-block", "width": "12px", "height": "12px",
            "borderRadius": "3px", "marginRight": "8px", "flex": "0 0 auto",
            "background": colour if on else "transparent",
            "border": f"2px solid {colour}", "opacity": 1 if on else 0.45})
        label = html.Span([
            swatch,
            html.Span(p, style={"color": t["text"] if on else t["faint"],
                                "fontWeight": 600 if on else 400,
                                "textDecoration": "none" if on else "line-through"})],
            title=PARTY_NAMES.get(p, p),
            style={"display": "inline-flex", "alignItems": "center",
                   "verticalAlign": "middle"})
        opts.append({"label": label, "value": p})
    return opts

# ===========================================================================
# INTERACTION LOGIC
#
# Pure functions extracted from the callbacks below. They take values and
# return values, touching no framework state, which is what allows the most
# intricate control logic in the application to be exercised by the test
# suite without a browser or a running server.
# ===========================================================================


def _norm(bound):
    return pd.to_datetime(bound).isoformat()


def parse_range(relayout):
    if not relayout:
        return None
    if "xaxis.range" in relayout:
        r = relayout["xaxis.range"]
        return [_norm(r[0]), _norm(r[1])]
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        return [_norm(relayout["xaxis.range[0]"]), _norm(relayout["xaxis.range[1]"])]
    return None


def decide_range(triggered_id, relayout, current):
    if triggered_id == "reset-btn":
        return None if current is not None else "keep"
    if relayout and relayout.get("xaxis.autorange"):
        return None if current is not None else "keep"
    new = parse_range(relayout)
    if new is None:
        return "keep"
    if current and current[0] == new[0] and current[1] == new[1]:
        return "keep"
    return new


def _iso_date(value):
    """'2025-03-04' (or a full ISO timestamp) -> ISO timestamp, else None."""
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else ts.isoformat()


def build_readout(hover, trends_records, parties, pollsters=None, theme="light"):
    t = T(theme)
    if not hover or not hover.get("points"):
        return html.Div("Hover over the chart to view data for any date.", 
                        style={"color": t["faint"], "fontStyle": "italic", "padding": "10px"})

    pt = hover["points"][0]
    cursor_date = pd.to_datetime(pt["x"]).normalize()
    cursor_ord = cursor_date.toordinal()

    out = [html.Div(cursor_date.strftime('%A, %d %B %Y'), 
                    style={"fontWeight": 700, "fontSize": "1.1em", "marginBottom": "12px"})]

    # interpolate trend for each party
    if trends_records and parties:
        tdf = pd.DataFrame(trends_records)
        tdf["date"] = pd.to_datetime(tdf["date"])
        tdf["ordinal"] = tdf["date"].apply(lambda d: d.toordinal())

        agg_spans = []
        for party in parties:
            sub = tdf[tdf["party"] == party].sort_values("ordinal")
            if sub.empty:
                continue
            # interpolate
            x_vals = sub["ordinal"].values
            y_vals = sub["trend"].values
            if cursor_ord < x_vals.min() or cursor_ord > x_vals.max():
                # fallback to nearest
                idx = (np.abs(x_vals - cursor_ord)).argmin()
                trend_val = y_vals[idx]
            else:
                trend_val = np.interp(cursor_ord, x_vals, y_vals)

            agg_spans.append(
                html.Span(f"{party} {trend_val:.1f}%",
                          style={"color": party_colour(party, theme), "fontWeight": 600,
                                 "marginRight": "16px", "display": "inline-block"})
            )
        out.append(html.Div(agg_spans, style={"marginBottom": "10px"}))

    out.append(html.Hr(style={"margin": "12px 0", "borderTop": "1px solid #ddd"}))

    # polls active on this date (respecting  pollster filter)
    source = filter_pollster(polls, pollsters)
    s = pd.to_datetime(source["fieldwork_start"]).dt.normalize()
    e = pd.to_datetime(source["fieldwork_end"]).dt.normalize()
    active_polls = source[(s <= cursor_date) & (e >= cursor_date)]

    if active_polls.empty:
        out.append(html.Div("No polls in fieldwork on this date.", style={"color": t["faint"], "fontSize": "0.9em"}))
    else:
        out.append(html.Div(f"{len(active_polls)} poll(s) active on this date:", 
                            style={"fontWeight": 600, "color": t["muted"], "marginBottom": "8px", "fontSize": "0.9em"}))
        
        # build a clean block for each overlapping poll
        for _, r in active_polls.sort_values("fieldwork_end").iterrows():
            fs_str = pd.to_datetime(r["fieldwork_start"]).strftime("%d %b")
            fe_str = pd.to_datetime(r["fieldwork_end"]).strftime("%d %b")
            
            # format results row for this specific poll
            poll_results = []
            for p in ALL_PARTIES:
                if p in r and pd.notna(r[p]):
                    poll_results.append(
                        html.Span(f"{p} {r[p]:.0f}%", 
                                  style={"color": party_colour(p, theme), "marginRight": "10px", "fontSize": "0.9em"})
                    )
            
            poll_block = html.Div([
                html.Strong(r['pollster'], style={"fontSize": "0.95em"}),
                html.Span(f" ({fs_str} to {fe_str} · n={fmt_n(r['sample_size'])})", 
                          style={"color": t["muted"], "fontSize": "0.85em"}),
                html.Div(poll_results, style={"marginTop": "1px"})
            ], style={"borderLeft": "3px solid #ccc", "paddingLeft": "10px", "marginBottom": "8px"})
            
            out.append(poll_block)
            
    return out


def cached_trends(windowed, parties, cache_key):
    """compute_trends is the expensive analytical step and is re-run on every
    interaction, yet its inputs change only when the window, party selection
    or pollster filter changes. Results are memoised on exactly those keys."""
    if cache_key in _TRENDS_CACHE:
        return _TRENDS_CACHE[cache_key]
    result = compute_trends(windowed, parties=parties)
    if len(_TRENDS_CACHE) > 64:          # keep the cache bounded
        _TRENDS_CACHE.clear()
    _TRENDS_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------


def render_coverage(windowed, parties, pollsters, theme="light"):
    """Explain any party drawn without a trend line.

    Insufficient evidence and absent support look identical on a chart: in
    both cases no line appears. Filtering to one polling organisation makes
    the first case common, since many published only a handful of national
    polls or began operating partway through the record. Saying so removes
    an ambiguity the visualisation cannot resolve on its own.
    """
    t = T(theme)
    if not parties or not len(windowed):
        return ""
    cover = trend_coverage(windowed, parties)
    thin = {p: c for p, c in cover.items() if not c["fitted"]}
    if not thin:
        return ""

    who = "these pollsters" if pollsters else "this selection"
    items = []
    for p, c in thin.items():
        if c["polls"] == 0:
            detail = f"not polled by {who} in this period"
        elif c["polls"] == 1:
            detail = "1 poll — too few to fit a trend"
        else:
            span = ""
            if c["first"] is not None and c["last"] is not None:
                span = (f", {c['first']:%b %Y}"
                        + (f" to {c['last']:%b %Y}"
                           if c["last"] != c["first"] else ""))
            detail = (f"{c['polls']} polls{span} — fewer than the "
                      f"{MIN_POLLS_FOR_TREND} needed to fit a trend")
        items.append(html.Div([
            html.Span(p, style={"color": party_colour(p, theme),
                                "fontWeight": 600, "marginRight": "6px"}),
            html.Span(detail, style={"color": t["muted"]})],
            style={"fontSize": "var(--fs-sm)", "padding": "1px 0"}))

    return html.Div(
        [html.Div("No trend shown for", className="panel-label")] + items
        + [html.Div("Individual polls are still drawn where they exist. An "
                    "absent line means insufficient evidence, not an absence "
                    "of support.",
                    style={"color": t["faint"], "fontSize": "var(--fs-xs)",
                           "fontStyle": "italic", "marginTop": "6px"})],
        style={"marginTop": "8px"})


def _placeholder(title, note=""):
    return html.Div([html.Div(title, className="placeholder-title"),
                     html.Div(note or "Not yet implemented")],
                    className="placeholder")


app.layout = html.Div([

    # shown instead of the dashboard below 900px (CSS media query): the design
    # brief rejects a reflowed mobile layout in favour of a courteous redirect.
    html.Div([
        html.Div("Pollster needs a wider screen",
                 style={"fontWeight": 700, "marginBottom": "8px"}),
        html.Div("This dashboard is built for desktop and tablet displays. "
                 "Please revisit on a device at least 900px wide.")],
        id="small-screen-notice"),

    html.Div([
        # ---- TOP STRIP ----
        html.Div([
            html.Div([
                html.H2("Pollster", className="app-title"),
                html.Span("UK voting-intention poll-of-polls — drag the timeline "
                          "to zoom; hover to read a date; click to pin.",
                          className="app-subtitle"),
                # FR7: the currency of the data is stated where a reader will
                # see it before drawing conclusions from the chart, rather
                # than being available only on inspection of the database.
                html.Span(
                    ["Data current as of ",
                     html.Strong(f"{DATA_AS_OF:%d %B %Y}"
                                 if DATA_AS_OF is not None else "unknown"),
                     " · sourced from ",
                     *[html.A("Wikipedia", href=u, target="_blank",
                              rel="noopener noreferrer",
                              style={"color": "inherit"})
                       for u in SOURCE_URLS[:1]],
                     (f" and {len(SOURCE_URLS) - 1} further page(s)"
                      if len(SOURCE_URLS) > 1 else "")],
                    className="app-subtitle", style={"marginTop": "2px"})],
                style={"display": "flex", "flexDirection": "column"}),
            html.Div([
                _placeholder("Poll topic selector", "Westminster VI only"),
                html.Div([
                    html.Button("⚙ Settings", id="settings-btn", n_clicks=0,
                                className="btn"),
                    html.Div([
                        html.Button("✕", id="settings-close", n_clicks=0,
                                    className="close-x"),
                        html.Div("Appearance", className="panel-label"),
                        dcc.RadioItems(
                            id="theme-picker",
                            options=[{"label": " Light", "value": "light"},
                                     {"label": " Dark", "value": "dark"}],
                            value="light",
                            labelStyle={"display": "block", "padding": "3px 0",
                                        "fontSize": "var(--fs-md)",
                                        "color": "var(--text)",
                                        "cursor": "pointer"})],
                        id="settings-panel", className="settings-panel",
                        style={"display": "none"})],
                    className="settings-wrap")],
                style={"display": "flex", "gap": "8px", "alignItems": "center"}),
            _placeholder("Learn more", "Methodology")],
                # settings: a button that opens a closable popover panel.
            style={"gridArea": "topbar", "display": "flex",
                   "justifyContent": "space-between", "alignItems": "center",
                   "gap": "16px", "flexWrap": "wrap",
                   "borderBottom": "1px solid var(--border-soft)",
                   "paddingBottom": "12px"}),

        # ---- LEFT RAIL ----
        html.Div([
            html.Div([
                html.Div([
                    html.Div("Isolate by party", className="panel-label",
                             style={"marginBottom": 0}),
                    html.Button("None", id="party-toggle-all", n_clicks=0,
                                className="btn btn--mini")],
                    style={"display": "flex", "justifyContent": "space-between",
                           "alignItems": "center", "marginBottom": "8px",
                           "flexWrap": "nowrap", "gap": "8px"}),
                dcc.Checklist(id="party-picker",
                              options=party_options(ALL_PARTIES),
                              value=ALL_PARTIES,
                              className="party-list",
                              labelStyle={"display": "flex", "alignItems": "center",
                                          "padding": "4px 2px", "cursor": "pointer"},
                              inputStyle={"position": "absolute", "opacity": 0,
                                          "width": "1px", "height": "1px",
                                          "margin": 0, "cursor": "pointer"})],
                className="panel"),
            html.Div([
                html.Div("Pollster filter", className="panel-label"),
                dcc.Dropdown(id="pollster-picker",
                             options=[{"label": p, "value": p} for p in POLLSTERS],
                             value=[], multi=True,
                             placeholder="All pollsters")],
                className="panel", style={"marginTop": "8px"})],
            style={"gridArea": "left"}),

        # ---- CENTRE ----
        html.Div([
            html.Div(dcc.Graph(id="trend-graph", style={"height": "62vh"},
                               config={"responsive": True}),
                     className="panel", style={"padding": "6px"}),
            # date boxes flank timeline: dragging remains available for
            # quick exploration, while typed dates give precision that a
            # multi-year drag cannot (both write to the same range-store)
            # as date range increases
            html.Div([
                html.Div([
                    html.Span("From", className="panel-label",
                              style={"marginBottom": 0}),
                    dcc.DatePickerSingle(
                        id="date-start", date=FULL_START[:10],
                        min_date_allowed=FULL_START[:10],
                        max_date_allowed=FULL_END[:10],
                        display_format="DD MMM YYYY",
                        className="date-box")],
                    style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                html.Button("Reset view", id="reset-btn", n_clicks=0,
                            className="btn"),
                html.Div([
                    html.Span("To", className="panel-label",
                              style={"marginBottom": 0}),
                    dcc.DatePickerSingle(
                        id="date-end", date=FULL_END[:10],
                        min_date_allowed=FULL_START[:10],
                        max_date_allowed=FULL_END[:10],
                        display_format="DD MMM YYYY",
                        className="date-box")],
                    style={"display": "flex", "alignItems": "center", "gap": "6px"})],
                style={"display": "flex", "alignItems": "center",
                       "justifyContent": "space-between", "margin": "10px 0"}),
            html.Div(id="hover-readout", className="panel panel--alt",
                     style={"minHeight": "3.2em", "marginBottom": "8px"}),
            html.Div(id="poll-detail", className="panel"),
            # Explains any party shown without a trend line — see
            # render_coverage. Empty, and so invisible, when every selected
            # party has enough evidence to be fitted.
            html.Div(id="coverage-note")],
            style={"gridArea": "center", "minWidth": 0}),

        # ---- RIGHT COLUMN ----
        html.Div([
            _placeholder("News today",
                         "Extractive summaries of the day's coverage (deferred)."),
            html.Div([
                html.Div("Net change", className="panel-label"),
                html.Div([
                    html.Span("over ", className="meta",
                              style={"marginRight": "6px"}),
                    dcc.Dropdown(id="netchange-anchor",
                                 options=netchange_options(events),
                                 value=NETCHANGE_DEFAULT, clearable=False,
                                 style={"width": "130px"})],
                    style={"display": "flex", "alignItems": "center",
                           "marginBottom": "10px"}),
                html.Div(id="netchange-panel")],
                className="panel", style={"marginTop": "8px"})],
            style={"gridArea": "right"}),
    ],
        id="dashboard",
        style={"display": "grid", "gap": "10px",
               "gridTemplateColumns": "230px minmax(0, 1fr) 280px",
               "gridTemplateAreas": '"topbar topbar topbar" "left center right"',
               "alignItems": "start"}),

    dcc.Store(id="range-store"),
    dcc.Store(id="pinned-store"),
    dcc.Store(id="trends-store"),
],
    id="app-root", className="app-root theme-light")


# ===========================================================================
# CALLBACKS
#
# Wiring only: each declares its inputs and outputs and delegates to the
# functions above. Keeping them together, and keeping them thin, is what
# makes the dependency graph readable in one place.
# ===========================================================================


@app.callback(Output("app-root", "className"),
              Input("theme-picker", "value"))
def apply_theme(theme):
    """swap the wrapper class; every CSS variable keys off it. Done server-side
    (rather than a clientside JS callback) to keep one mechanism for theming."""
    return f"app-root theme-{theme or 'light'}"


@app.callback(Output("settings-panel", "style"),
              Input("settings-btn", "n_clicks"),
              Input("settings-close", "n_clicks"),
              State("settings-panel", "style"),
              prevent_initial_call=True)
def toggle_settings(_open, _close, style):
    """Settings is a closable popover: the button opens it, ✕ closes it."""
    if ctx.triggered_id == "settings-close":
        return {"display": "none"}
    showing = (style or {}).get("display") != "none"
    return {"display": "none"} if showing else {"display": "block"}

@app.callback(Output("netchange-panel", "children"),
              Input("netchange-anchor", "value"),
              Input("theme-picker", "value"),
              State("trends-store", "data"))
def update_netchange(anchor_value, theme, trends_records):
    result = compute_net_change(trends_records or [], anchor_value,
                                ALL_PARTIES, FULL_START)
    return render_netchange(result, theme)

@app.callback(Output("range-store", "data"),
              Output("date-start", "date"),
              Output("date-end", "date"),
              Input("trend-graph", "relayoutData"),
              Input("reset-btn", "n_clicks"),
              Input("date-start", "date"),
              Input("date-end", "date"),
              State("range-store", "data"))
def update_range(relayout, reset_clicks, start_value, end_value, current):
    """single owner of the visible range, whichever control set it.

    the range slider & the two date boxes are two views of one value, so
    they must not each own it: this callback takes whichever fired and
    writes back to BOTH, which keeps them in step without a feedback loop
    (Dash only re-fires when a value actually changes).
    """
    trigger = ctx.triggered_id

    if trigger in ("date-start", "date-end"):
        # typed dates: fall back to the full extent for a blank/invalid box,
        # and swap them if they were entered the wrong way round.
        lo = _iso_date(start_value) or FULL_START
        hi = _iso_date(end_value) or FULL_END
        if lo > hi:
            lo, hi = hi, lo
        return [lo, hi], lo[:10], hi[:10]

    result = decide_range(trigger, relayout, current)
    if result == "keep":
        return no_update, no_update, no_update
    # dragging or resetting: mirror the new range into the date boxes.
    lo, hi = (result if result else [FULL_START, FULL_END])
    return result, lo[:10], hi[:10]


@app.callback(Output("pinned-store", "data"), Output("poll-detail", "children"),
              Output("trend-graph", "figure", allow_duplicate=True),
              Input("trend-graph", "clickData"),
              State("pollster-picker", "value"),
              State("theme-picker", "value"),
              prevent_initial_call=True)
def pin_poll(click_data, pollsters, theme):
    t = T(theme)
    """a click pins a DATE: the code then lists every poll in fieldwork on
    that date, so overlapping polls all appear (not just one). respects the
    pollster filter so the pinned list matches what's on screen. the pin line
    is drawn by patching shape[0]/annotation[0] only - no full figure is rebuilt."""
    if not click_data or not click_data.get("points"):
        return no_update, no_update, no_update
    x = click_data["points"][0].get("x")
    if x is None:
        return no_update, no_update, no_update
    D = pd.to_datetime(x).normalize()

    source = filter_pollster(polls, pollsters)
    s = pd.to_datetime(source["fieldwork_start"]).dt.normalize()
    e = pd.to_datetime(source["fieldwork_end"]).dt.normalize()
    active = source[(s <= D) & (e >= D)]

    out = [html.Strong(f"📌 Pinned — {D:%A, %d %b %Y}"), html.Br(),
           html.Span(f"{len(active)} poll(s) in fieldwork on this date",
                     style={"color": t["muted"], "fontSize": "0.9em"})]
    for _, r in active.sort_values("fieldwork_end").iterrows():
        fs = pd.to_datetime(r["fieldwork_start"]).strftime("%d %b")
        fe = pd.to_datetime(r["fieldwork_end"]).strftime("%d %b")
        nums = [html.Span(f"{p} {r[p]:.0f}%",
                          style={"color": party_colour(p, theme),
                                 "marginRight": "10px", "fontSize": "0.9em"})
                for p in ALL_PARTIES if p in r and pd.notna(r[p])]
        # FR4: a poll's scope, method and origin belong beside its numbers,
        # where they inform the reading rather than sitting in the database
        # unread. Each is omitted rather than shown blank when absent.
        area = r.get("area")
        method = r.get("methodology_type")
        src = r.get("source_url")
        meta = f" ({fs} to {fe} · n={fmt_n(r['sample_size'])}"
        if pd.notna(area):
            meta += f" · {area}"
        if pd.notna(method) and str(method).lower() != "standard":
            meta += f" · {method}"
        meta += ")"
        out.append(html.Div([
            html.Strong(r["pollster"], style={"fontSize": "0.95em"}),
            html.Span(meta,
                      style={"color": t["muted"], "fontSize": "0.85em"}),
            (html.Span([" · ", html.A("source", href=src, target="_blank",
                                      rel="noopener noreferrer",
                                      style={"color": t["muted"]})],
                       style={"fontSize": "0.85em"})
             if pd.notna(src) else ""),
            html.Div(nums, style={"marginTop": "1px"})],
            style={"borderLeft": f"3px solid {t['pin']}", "paddingLeft": "10px",
                   "marginTop": "8px"}))

    # patches only the reserved pin shape/annotation so the rest of the fig is
    # left untouched, this is near-instant vs prev full build_figure rebuild.
    iso = D.isoformat()
    patched = Patch()
    patched["layout"]["shapes"][0]["x0"] = iso
    patched["layout"]["shapes"][0]["x1"] = iso
    patched["layout"]["shapes"][0]["visible"] = True
    patched["layout"]["annotations"][0]["x"] = iso
    patched["layout"]["annotations"][0]["visible"] = True
    return {"date": iso}, out, patched


import numpy as np

@app.callback(Output("hover-readout", "children"),
              Input("trend-graph", "hoverData"),
              State("trends-store", "data"), State("party-picker", "value"),
              State("pollster-picker", "value"),
              State("theme-picker", "value"))
def show_readout(hover, trends_records, parties, pollsters, theme):
    return build_readout(hover, trends_records, parties, pollsters, theme)


_TRENDS_CACHE = {}


@app.callback(Output("trend-graph", "figure"), Output("trends-store", "data"),
              Output("coverage-note", "children"),
              Input("party-picker", "value"),
              Input("range-store", "data"),
              Input("pollster-picker", "value"),
              Input("theme-picker", "value"),
              State("pinned-store", "data"))
def update_graph(selected_parties, range_iso, pollsters, theme, pinned):
    selected_parties = selected_parties or []
    subset = filter_pollster(polls, pollsters)
    windowed = filter_range(subset, range_iso)
    key = (str(range_iso), tuple(selected_parties), tuple(pollsters or []))
    trends = cached_trends(windowed, selected_parties, key)
    records = ([] if not len(trends) else
               trends.assign(date=trends["date"].astype(str))
               [["date", "party", "trend"]].to_dict("records"))
    return (build_figure(subset, trends, parties=selected_parties,
                         window_range=range_iso, pinned=pinned, events=events,
                         theme=theme),
            records,
            render_coverage(windowed, selected_parties, pollsters, theme))

@app.callback(Output("party-picker", "options"),
              Output("party-picker", "value"),
              Output("party-toggle-all", "children"),
              Input("party-picker", "value"),
              Input("party-toggle-all", "n_clicks"),
              Input("theme-picker", "value"))
def restyle_parties(selected, _n, theme):
    """redrawsthe party rows so deselected ones are visibly muted (Dash has no
    per-option style hook that reacts to selection), and handle all/none."""
    if ctx.triggered_id == "party-toggle-all":
        selected = [] if (selected and len(selected) == len(ALL_PARTIES)) else list(ALL_PARTIES)
    selected = selected or []
    label = "None" if len(selected) == len(ALL_PARTIES) else "All"
    return party_options(selected, theme), selected, label

if __name__ == "__main__":
    app.run(debug=True)