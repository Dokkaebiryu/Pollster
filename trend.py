import numpy as np
import pandas as pd

# NOTE: analyser is imported inside main(), not here. This module's
# statistical functions take DataFrames and know nothing about storage;
# importing at module level would drag the database layer in behind them
# and make them untestable in isolation.

DEFAULT_PARTIES = ["Lab", "Con", "Ref", "LD", "Grn"]
EXPANDED_PARTIES = ["Lab", "Con", "Ref", "LD", "Grn", "SNP", "PC", "RB",
                    "UKIP", "Brx", "Chg", "Others"]
DEFAULT_SPAN = 0.3          # fraction of local data per fit (cf. 0.25-0.35)
WEIGHT_CAP_Q = 0.95
# No estimate is produced for a date with no poll within this many days.
# A local LINEAR fit evaluated far from its data extrapolates the local
# slope without limit: across a long gap the nearest polls all lie on one
# side, and the fitted line runs away from them. Filtering to a single
# pollster makes such gaps routine, and produced vote shares above 400%
# before this guard existed. Refusing to draw is the honest response -
# the absence of a line states the absence of evidence.
MAX_GAP_DAYS = 60

# fewer polls than this and no trend is fitted at all. several polling
# organisations in the record published only a handful of national polls,
# or began operating partway through the period; smoothing three
# observations would produce a confident-looking line resting on almost
# nothing.
MIN_POLLS_FOR_TREND = 5         # winsorise sample sizes at this quantile to avoid a few very large polls dominating the trend


def _tricube(u):
    u = np.clip(np.abs(u), 0.0, 1.0)
    return (1.0 - u**3) ** 3


def weighted_loess(x, y, n, grid, frac=DEFAULT_SPAN, max_gap=MAX_GAP_DAYS):
    """sample-size-weighted local linear regression.

    returns (trend, band_n): the smoothed values on `grid`, & the
    tricube-weighted MEAN local sample size used for the band (as a typical
    poll's precision near each point, rather than an over-precise pooled total).
    """
    x = np.asarray(x, float); y = np.asarray(y, float); n = np.asarray(n, float)
    grid = np.asarray(grid, float)
    trend = np.full(len(grid), np.nan)
    band_n = np.zeros(len(grid))
    k = max(3, int(np.ceil(frac * len(x))))

    for i, g in enumerate(grid):
        d = np.abs(x - g)
        # Local support test, applied before any fitting: with no poll close
        # by, whatever the regression returns is extrapolation rather than
        # an estimate, so nothing is returned at all.
        if max_gap is not None and d.min() > max_gap:
            continue                               # leaves NaN; the line breaks
        h = np.partition(d, k - 1)[k - 1]          # distance to k-th nearest
        if h <= 0:
            h = d.max() if d.max() > 0 else 1.0
        tri = _tricube(d / h)
        w = tri * n                                # locality x precision
        if w.sum() <= 0:
            continue
        dx = x - g
        Sw, Swx = w.sum(), (w * dx).sum()
        Swxx = (w * dx * dx).sum()
        Swy, Swxy = (w * y).sum(), (w * dx * y).sum()
        det = Sw * Swxx - Swx * Swx
        fitted = (Swxx * Swy - Swx * Swxy) / det if abs(det) > 1e-9 else Swy / Sw

        # A local fit may interpolate between nearby observations; it may not
        # invent values beyond them. Where the neighbourhood is small and its
        # slope steep, an unconstrained linear fit projects that slope past
        # the data it was fitted to, which is how a party polling at 48%
        # acquired a trend of 892%. Bounding the estimate by the local
        # observations keeps genuine movement while removing the artefact.
        local = y[tri > 0]
        if local.size:
            fitted = min(max(fitted, local.min()), local.max())
        trend[i] = fitted
        band_n[i] = (tri * n).sum() / tri.sum()    # weighted-mean local n

    return trend, band_n


def confidence_band(trend, band_n, z=1.96):
    """95% band from binomial sampling error at the local typical sample size.
    NOTE: captures sampling error only; 'house' effects & systematic error
    (Kellner pts 10, 13(3)) make true uncertainty wider - this is a lower bound."""
    p = np.clip(trend / 100.0, 0, 1)
    moe = z * np.sqrt(p * (1 - p) / np.maximum(band_n, 1)) * 100.0
    return trend - moe, trend + moe


def sample_size_report(df: pd.DataFrame) -> None:
    """Health check: how usable are the sample sizes we weight by?"""
    total = len(df)
    valid = df["sample_size"].notna() & (df["sample_size"] > 0)
    print(f"Sample sizes: {valid.sum()}/{total} usable "
          f"({total - valid.sum()} missing/invalid, excluded from weighting)")
    if valid.any():
        s = df.loc[valid, "sample_size"]
        print(f"  min {int(s.min())}, median {int(s.median())}, max {int(s.max())}")


def compute_trends(df: pd.DataFrame, parties=DEFAULT_PARTIES,
                   span=DEFAULT_SPAN, grid_days=200) -> pd.DataFrame:
    """tidy trend frame: date, party, trend, lo, hi.

    each party's trend spans only the range where that party actually has
    poll data, so a party isn't shown before it existed / was polled (e.g.
    RB), & values are clipped at zero since a vote share can't be negative.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["fieldwork_end"])
    df = df.dropna(subset=["date"])
    if len(df) == 0 or not parties:
        return pd.DataFrame(columns=["date", "party", "trend", "lo", "hi"])
    origin = df["date"].min()
    df["x"] = (df["date"] - origin).dt.days

    out = []
    for party in parties:
        if party not in df.columns:
            continue
        valid = df[party].notna() & df["sample_size"].notna() & (df["sample_size"] > 0)
        sub = df.loc[valid]
        if len(sub) < MIN_POLLS_FOR_TREND:
            continue
        # Grid restricted to THIS party's own data range (not the global span).
        grid_x = np.linspace(sub["x"].min(), sub["x"].max(), grid_days)
        grid_dates = origin + pd.to_timedelta(grid_x, unit="D")
        
        # MRP polls report enormous nominal n (40k+) but are modelled
        # constituency estimates, not simple random samples - their national
        # precision is nowhere near what n implies. The WEIGHT is capped (the
        # recorded sample_size is left intact for provenance) so that a single
        # MRP cannot dominate the local fit or falsely narrow the band.
        weights = sub["sample_size"].values.astype(float)
        if len(weights) > 1:
            weights = np.minimum(weights, np.quantile(weights, WEIGHT_CAP_Q))
            
        trend, band_n = weighted_loess(sub["x"].values, sub[party].values,
                                       weights, grid_x, span)
        lo, hi = confidence_band(trend, band_n)
        # Domain constraints the mathematics does not supply: a vote share
        # lies between 0 and 100. The upper bound matters as much as the
        # lower - an unconstrained local fit can exceed 100% as readily as
        # it can fall below zero.
        trend = np.clip(trend, 0, 100)
        lo = np.clip(lo, 0, 100)
        hi = np.clip(hi, 0, 100)
        out.append(pd.DataFrame({
            "date": grid_dates, "party": party,
            "trend": trend, "lo": lo, "hi": hi}))

    if not out:
        return pd.DataFrame(columns=["date", "party", "trend", "lo", "hi"])
    return pd.concat(out, ignore_index=True)


def trend_coverage(df: pd.DataFrame, parties) -> dict:
    """Why a party has no trend line, per party.

    The interface draws nothing where evidence is insufficient, which is
    correct but indistinguishable from a party having no support. This
    reports the reason so the interface can say it: how many usable polls
    exist, and the period they span.
    """
    out = {}
    if "fieldwork_end" not in df.columns:
        return out
    dates = pd.to_datetime(df["fieldwork_end"], errors="coerce")
    for party in parties:
        if party not in df.columns:
            out[party] = {"polls": 0, "first": None, "last": None,
                          "fitted": False}
            continue
        ok = (df[party].notna() & df["sample_size"].notna()
              & (df["sample_size"] > 0))
        n = int(ok.sum())
        sub_dates = dates[ok]
        out[party] = {
            "polls": n,
            "first": sub_dates.min() if n else None,
            "last": sub_dates.max() if n else None,
            "fitted": n >= MIN_POLLS_FOR_TREND,
        }
    return out


def _preview_html(df_polls, trends, parties=DEFAULT_PARTIES,
                  path="trend_preview.html"):
    """writes a quick Plotly preview so the span can be eyeballed.
    (only used as a development aid; the real chart is to be built in the Dash layer.)"""
    import plotly.graph_objects as go
    colours = {"Lab": "#E4003B", "Con": "#0087DC", "Ref": "#12B6CF",
               "LD": "#FF6400", "Grn": "#00A85A"}
    dfp = df_polls.copy()
    dfp["date"] = pd.to_datetime(dfp["fieldwork_end"])
    fig = go.Figure()
    for party in parties:
        c = colours.get(party, "#888")
        t = trends[trends["party"] == party]
        # scattered raw polls
        fig.add_trace(go.Scatter(x=dfp["date"], y=dfp[party], mode="markers",
                                 name=f"{party} polls", marker=dict(size=4, color=c),
                                 opacity=0.25, showlegend=False))
        # band
        fig.add_trace(go.Scatter(
            x=list(t["date"]) + list(t["date"][::-1]),
            y=list(t["hi"]) + list(t["lo"][::-1]),
            fill="toself", fillcolor=c, opacity=0.12,
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        # trend
        fig.add_trace(go.Scatter(x=t["date"], y=t["trend"], mode="lines",
                                 name=party, line=dict(color=c, width=2.5)))
    fig.update_layout(title="Pollster - weighted LOESS trend (span %.2f)" % DEFAULT_SPAN,
                      template="plotly_white", yaxis_title="Vote share (%)")
    fig.write_html(path)
    print(f"Wrote preview: {path}  (open it in browser)")


def main() -> None:
    from analyser import load_topic_polls          # see note at the imports

    df = load_topic_polls()
    print(f"Loaded {len(df)} polls\n")
    sample_size_report(df)
    print("\nComputing weighted trends...")
    trends = compute_trends(df)

    latest = trends.sort_values("date").groupby("party").tail(1)
    print("\nLatest smoothed estimate per party:")
    for _, r in latest.iterrows():
        print(f"  {r['party']:>4}: {r['trend']:5.1f}%  "
              f"(band {r['lo']:.1f}-{r['hi']:.1f})")

    _preview_html(df, trends)


if __name__ == "__main__":
    main()