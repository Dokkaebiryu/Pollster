import numpy as np
import pandas as pd
import pytest

import app


# filter_pollster (whole-view scoping)

@pytest.fixture
def polls_frame():
    return pd.DataFrame({
        "pollster": ["YouGov", "Opinium", "YouGov", "Savanta"],
        "sample_size": [2000, 1800, 2100, 1500],
        "Lab": [25.0, 26.0, 24.0, 27.0],
    })


def test_empty_selection_returns_every_pollster(polls_frame):
    """an empty filter means 'all', not 'none': the control is additive,
    an empty dropdown is its resting state rather."""
    assert len(app.filter_pollster(polls_frame, [])) == 4


def test_none_selection_returns_every_pollster(polls_frame):
    assert len(app.filter_pollster(polls_frame, None)) == 4


def test_single_selection_restricts_to_that_pollster(polls_frame):
    out = app.filter_pollster(polls_frame, ["YouGov"])
    assert len(out) == 2
    assert set(out["pollster"]) == {"YouGov"}


def test_multiple_selection_restricts_to_those_pollsters(polls_frame):
    out = app.filter_pollster(polls_frame, ["YouGov", "Savanta"])
    assert set(out["pollster"]) == {"YouGov", "Savanta"}


def test_unknown_pollster_yields_an_empty_frame_not_an_error(polls_frame):
    """a stale selection must not raise: the pollster list is rebuilt when
    the database is reloaded, and a previously valid name may vanish."""
    out = app.filter_pollster(polls_frame, ["NotAPollster"])
    assert len(out) == 0


# decide_range (feedback-loop breaker)

def test_a_new_range_from_a_drag_is_accepted():
    result = app.decide_range(
        "trend-graph",
        {"xaxis.range[0]": "2025-01-01", "xaxis.range[1]": "2025-06-01"},
        None)
    assert result != "keep" and result is not None
    assert result[0].startswith("2025-01-01")


def test_an_identical_range_is_rejected_to_break_the_loop():
    """the central defect this function exists to prevent: returning a figure
    re-emits relayoutData, which would re-invoke the callback indefinitely
    unless a repeat of the current range is recognised and ignored."""
    current = ["2025-01-01T00:00:00", "2025-06-01T00:00:00"]
    result = app.decide_range(
        "trend-graph",
        {"xaxis.range[0]": "2025-01-01", "xaxis.range[1]": "2025-06-01"},
        current)
    assert result == "keep"


def test_both_relayout_shapes_are_understood():
    """plotly reports the range either as a pair of indexed keys or as a
    single list, depending on interaction."""
    indexed = app.parse_range({"xaxis.range[0]": "2025-01-01",
                               "xaxis.range[1]": "2025-06-01"})
    listed = app.parse_range({"xaxis.range": ["2025-01-01", "2025-06-01"]})
    assert indexed == listed


def test_reset_clears_an_active_range():
    assert app.decide_range("reset-btn", None, ["2025-01-01", "2025-06-01"]) is None


def test_reset_when_already_full_makes_no_change():
    """returning None again would rewrite the store with an identical value
    and re-trigger the graph without reason."""
    assert app.decide_range("reset-btn", None, None) == "keep"


def test_plotly_autorange_is_treated_as_a_reset():
    """double-clicking the chart autoranges it, which must clear the stored
    window."""
    assert app.decide_range("trend-graph", {"xaxis.autorange": True},
                            ["2025-01-01", "2025-06-01"]) is None


def test_irrelevant_relayout_events_are_ignored():
    """plotly emits relayoutData for interactions unrelated to the x range,
    such as legend or drag-mode changes."""
    assert app.decide_range("trend-graph", {"dragmode": "zoom"}, None) == "keep"
    assert app.decide_range("trend-graph", None, None) == "keep"


# _iso_date (date-picker input handling)

def test_plain_dates_and_timestamps_both_resolve():
    assert app._iso_date("2025-03-04").startswith("2025-03-04")
    assert app._iso_date("2025-03-04T12:00:00").startswith("2025-03-04")


def test_blank_and_invalid_dates_resolve_to_none():
    assert app._iso_date(None) is None
    assert app._iso_date("") is None
    assert app._iso_date("not a date") is None


# resolve_anchor (net-change baselines)

def test_relative_anchors_count_back_from_the_supplied_end_date():
    end = pd.Timestamp("2026-07-06")
    start, label = app.resolve_anchor("rel:30", end)
    assert start == end - pd.Timedelta(days=30)
    assert "30" in label


def test_a_year_anchor_uses_its_human_label():
    _start, label = app.resolve_anchor("rel:365", pd.Timestamp("2026-07-06"))
    assert label == "1 year"


def test_an_absent_anchor_falls_back_to_the_default():
    end = pd.Timestamp("2026-07-06")
    start, _label = app.resolve_anchor(None, end)
    assert start == end - pd.Timedelta(days=30)


# compute_net_change (reporting, and refusing to report)

def _records(parties=("Lab", "Con"), start="2025-01-01", days=400,
             values=(20.0, 30.0)):
    """Trend records as the store holds them: date, party, trend."""
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for party, v in zip(parties, values):
        for i, d in enumerate(dates):
            rows.append({"date": str(d), "party": party,
                         "trend": v + i * 0.01})
    return rows


def test_net_change_reports_movement_between_two_dates():
    result = app.compute_net_change(_records(), "rel:30", ["Lab", "Con"])
    assert result["ok"]
    assert result["end"] > result["start"]
    deltas = {p: v for p, kind, v in result["rows"] if kind == "delta"}
    assert set(deltas) == {"Lab", "Con"}
    assert all(v > 0 for v in deltas.values())      # series is rising


def test_the_compared_dates_are_returned_for_display():
    """a signed figure w/o its interval is uninterpretable, so the panel
    must always be able to state what it compared."""
    result = app.compute_net_change(_records(), "rel:90", ["Lab"])
    assert (result["end"] - result["start"]).days == pytest.approx(90, abs=1)


def test_a_window_too_narrow_for_the_period_refuses_and_explains():
    """2 points closer together than the smoothing bandwidth differ mainly
    because of the smoothing, so the panel declines rather than reporting an
    artefact of smoothing."""
    result = app.compute_net_change(_records(days=10), "rel:90", ["Lab"])
    assert not result["ok"]
    assert "90" in result["message"] or "day" in result["message"].lower()


def test_no_trend_in_view_refuses_and_explains():
    result = app.compute_net_change([], "rel:30", ["Lab"])
    assert not result["ok"]
    assert result["message"]


def test_a_party_postdating_the_anchor_is_marked_new_not_changed():
    """reporting a numeric change for a party that did not exist at the
    anchor would be a fabricated comparison.

    both series must end on the same date, or the earlier-ending party is
    correctly reported as lacking recent data and the intended distinction
    is not the one under test.
    """
    records = _records(parties=["Lab"], start="2025-01-01", days=400)
    last = pd.Timestamp("2025-01-01") + pd.Timedelta(days=399)
    rb_start = pd.Timestamp("2026-01-01")
    rb_days = (last - rb_start).days + 1
    late = _records(parties=["RB"], start=str(rb_start.date()),
                    days=rb_days, values=[5.0])

    result = app.compute_net_change(records + late, "rel:365", ["Lab", "RB"])
    assert result["ok"]
    kinds = {p: kind for p, kind, _v in result["rows"]}
    assert kinds["Lab"] == "delta"
    assert kinds["RB"] == "new"


def test_only_selected_parties_are_reported_and_in_order():
    result = app.compute_net_change(
        _records(parties=("Lab", "Con", "Ref"), values=(20.0, 30.0, 25.0)),
        "rel:30", ["Con", "Lab"])
    assert [p for p, _k, _v in result["rows"]] == ["Con", "Lab"]


# netchange_options (control's own contents)

def test_relative_periods_are_offered_and_short_ones_are_not():
    """periods below smoothing bandwidth are withheld: the
    control does not offer a choice it would have to disclaim."""
    values = [o["value"] for o in app.netchange_options(app.events)]
    assert "rel:30" in values
    assert "rel:365" in values
    assert not any(v.startswith("rel:") and int(v.split(":")[1]) < 30
                   for v in values)


# party_options (redundant coding of selection state)

def test_selection_state_is_encoded_by_more_than_colour():
    """accessibility requirement NFR5: a user who cannot distinguish the
    swatch colours must still be able to tell selected from deselected."""
    opts = app.party_options(["Lab"])
    by_value = {o["value"]: o["label"] for o in opts}

    selected_text = by_value["Lab"].children[1].style
    deselected_text = by_value["Con"].children[1].style
    selected_swatch = by_value["Lab"].children[0].style
    deselected_swatch = by_value["Con"].children[0].style

    assert selected_text["fontWeight"] != deselected_text["fontWeight"]
    assert selected_text["textDecoration"] != deselected_text["textDecoration"]
    assert selected_swatch["background"] != deselected_swatch["background"]
    assert selected_swatch["opacity"] != deselected_swatch["opacity"]


def test_every_party_carries_its_full_name_for_readers_who_need_it():
    opts = app.party_options(app.ALL_PARTIES)
    for o in opts:
        assert o["label"].title
        assert o["label"].title != o["value"]      # a name, not the code