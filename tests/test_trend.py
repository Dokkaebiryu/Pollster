import numpy as np
import pandas as pd
import pytest

import trend


def _synthetic(n=200, start="2024-01-01", slope=0.02, base=30.0,
               noise=1.0, sample=1500, seed=0):
    """a series w a known underlying trend and added noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="3D")
    x = np.arange(n)
    truth = base + slope * x
    return pd.DataFrame({
        "fieldwork_end": dates,
        "pollster": ["Fixture"] * n,
        "sample_size": [sample] * n,
        "Lab": truth + rng.normal(0, noise, n),
    }), truth


# smoother recovers a known signal

def test_smoother_recovers_a_known_linear_trend():
    """the validation described in the Realisation: a signal of
    known shape, obscured by noise, should be recovered closely. this tests
    the implementation, rather than span selection."""
    df, truth = _synthetic()
    out = trend.compute_trends(df, parties=["Lab"], grid_days=len(df))
    fitted = out["trend"].values
    # compare on the interior, where a local fit is not boundary-affected
    interior = slice(len(fitted) // 10, -len(fitted) // 10)
    error = np.abs(fitted[interior] - truth[interior])
    assert error.mean() < 0.5


def test_smoother_output_is_smoother_than_its_input():
    """a smoother that does not reduce point-to-point variation is not
    smoothing."""
    df, _truth = _synthetic(noise=3.0)
    out = trend.compute_trends(df, parties=["Lab"], grid_days=len(df))
    raw_variation = np.abs(np.diff(df["Lab"].values)).mean()
    fitted_variation = np.abs(np.diff(out["trend"].values)).mean()
    assert fitted_variation < raw_variation


# weighting behaves as intended

def test_larger_samples_pull_the_fit_towards_themselves():
    """sample-size weighting is the project's central analytical decision;
    if a large poll did not pull the estimate, the weighting would be
    decorative.

    note sample size of the test itself: w very few points the
    tricube kernel assigns zero weight to the outermost, so weighting
    cannot be observed at all. Enough points are used that the kernel is
    not the dominant effect.
    """
    n = 41
    x = np.arange(float(n))
    y = np.full(n, 30.0)
    y[n // 2] = 40.0                       # one dissenting poll, centrally placed
    grid = np.array([float(n // 2)])

    equal, _ = trend.weighted_loess(x, y, np.full(n, 1000.0), grid)
    heavy_n = np.full(n, 1000.0)
    heavy_n[n // 2] = 50000.0              # the dissenter is a very large poll
    heavy, _ = trend.weighted_loess(x, y, heavy_n, grid)

    assert heavy[0] > equal[0]
    assert equal[0] < 40.0                 # one poll does not carry the fit
    assert heavy[0] < 40.0                 # nor does it even when large


def test_band_reflects_local_sample_size_not_the_pooled_total():
    """deriving the band from the pooled sample would imply a precision
    polling lacks. doubling every poll's size should narrow the
    band, but only as the square root of that change."""
    _t, band_small = trend.weighted_loess(
        np.arange(20.0), np.full(20, 30.0), np.full(20, 1000.0),
        np.array([10.0]))
    _t2, band_large = trend.weighted_loess(
        np.arange(20.0), np.full(20, 30.0), np.full(20, 4000.0),
        np.array([10.0]))
    assert band_small[0] == pytest.approx(1000, rel=0.01)
    assert band_large[0] == pytest.approx(4000, rel=0.01)


def test_confidence_band_width_is_plausible_for_polling():
    """a band far narrower than the observed scatter of polls would be
    self-evidently wrong, whatever its derivation."""
    trend_vals = np.array([30.0])
    lo, hi = trend.confidence_band(trend_vals, np.array([1500.0]))
    width = hi[0] - lo[0]
    assert 1.5 < width < 6.0


# domain constraints the mathematics doesnt supply

def test_a_party_is_not_drawn_before_it_has_data():
    """a local fit has no knowledge of what it is fitting; w/o this
    constraint a party is extrapolated back through years in which it did
    not exist."""
    df, _ = _synthetic(n=120)
    df["RB"] = np.nan
    df.loc[df.index[-40:], "RB"] = 5.0          # exists only recently
    out = trend.compute_trends(df, parties=["Lab", "RB"])
    rb_start = out[out["party"] == "RB"]["date"].min()
    lab_start = out[out["party"] == "Lab"]["date"].min()
    assert rb_start > lab_start
    assert rb_start >= df.loc[df.index[-40], "fieldwork_end"]


def test_vote_share_is_never_negative():
    """a linear local fit is unconstrained and may extrapolate below zero,
    which is not a quantity that can exist."""
    df, _ = _synthetic(n=120, base=2.0, slope=-0.02, noise=0.8)
    out = trend.compute_trends(df, parties=["Lab"])
    assert (out["trend"] >= 0).all()
    assert (out["lo"] >= 0).all()


def test_not_prompted_values_are_excluded_rather_than_treated_as_zero():
    """the decision made in cleaning must still hold: a null must not
    drag the fitted line towards zero."""
    df, _ = _synthetic(n=120)
    with_nulls = df.copy()
    with_nulls.loc[with_nulls.index[::3], "Lab"] = np.nan
    out_full = trend.compute_trends(df, parties=["Lab"])
    out_nulls = trend.compute_trends(with_nulls, parties=["Lab"])
    assert out_nulls["trend"].mean() == pytest.approx(
        out_full["trend"].mean(), abs=1.0)


# robustness to real but awkward inputs

def test_no_parties_selected_returns_an_empty_frame_not_an_error():
    """deselecting every party is a reasonable user action, not a fault."""
    df, _ = _synthetic()
    out = trend.compute_trends(df, parties=[])
    assert out.empty
    assert list(out.columns) == ["date", "party", "trend", "lo", "hi"]


def test_too_few_polls_to_fit_returns_an_empty_frame():
    df, _ = _synthetic(n=3)
    out = trend.compute_trends(df, parties=["Lab"])
    assert out.empty


def test_a_party_absent_from_the_data_is_skipped_silently():
    df, _ = _synthetic()
    out = trend.compute_trends(df, parties=["Lab", "NotAParty"])
    assert set(out["party"].unique()) == {"Lab"}


def test_polls_without_usable_sample_sizes_are_excluded_from_weighting():
    """the weighting scheme depends on a field the source may not populate."""
    df, _ = _synthetic(n=60)
    df.loc[df.index[:10], "sample_size"] = np.nan
    out = trend.compute_trends(df, parties=["Lab"])
    assert not out.empty
    assert out["trend"].notna().all()