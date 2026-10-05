from io import StringIO

import numpy as np
import pandas as pd
import pytest

import cleaner


# selection: hypothetical-scenario hazard

def test_selection_takes_only_wanted_year_headings(page_html):
    tables = cleaner.select_national_tables(page_html, ["2026", "2025"])
    assert len(tables) == 2


def test_selection_excludes_structurally_identical_hypothetical_tables(page_html):
    """the central hazard: hypothetical-scenario tables share the real
    tables' column signature exactly, so any content-based filter admits
    them. only heading context separates the two."""
    tables = cleaner.select_national_tables(page_html, ["2026", "2025"])
    combined = pd.concat(tables, ignore_index=True)
    flat = cleaner.flatten_headers(combined)
    assert "ShouldNotAppear" not in flat["Pollster"].astype(str).values


def test_selection_tags_each_table_with_its_source_year():
    """the year exists only in the heading, as the source writes year-less.
    discarding it at selection would make dates unparseable."""
    html = "<h2>2026</h2><table><tr><th>Pollster</th></tr>" \
           "<tr><td>YouGov</td></tr></table>"
    tables = cleaner.select_national_tables(html, ["2026"])
    assert tables[0]["source_year"].iloc[0] == "2026"


def test_selection_raises_when_no_heading_matches(page_html):
    """failing loudly is better thanreturning the wrong tables: the
    source is edited by third parties and headings may change."""
    with pytest.raises(ValueError):
        cleaner.select_national_tables(page_html, ["1999"])

# header handling

def test_flatten_headers_keeps_top_level_name():
    df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
        [("Lab", "Unnamed: 5_level_1"), ("Con", "Unnamed: 6_level_1")]))
    assert list(cleaner.flatten_headers(df).columns) == ["Lab", "Con"]


def test_flatten_headers_tolerates_single_level_headers():
    """older source pages do not consistently use 2 header rows."""
    df = pd.DataFrame(columns=["Lab", "Con"])
    assert list(cleaner.flatten_headers(df).columns) == ["Lab", "Con"]


def test_column_names_are_canonicalised_across_sources(alt_header_page):
    """'dates conducted' and 'date(s) conducted' are the same column"""
    tables = cleaner.select_national_tables(alt_header_page, ["2023"])
    df = cleaner.canonicalise_columns(cleaner.flatten_headers(tables[0]))
    assert "Date(s) conducted" in df.columns


# values: relating to not-prompted decision

def test_percentages_are_converted_to_numbers():
    df = pd.DataFrame({"Lab": ["25%"], "Con": ["24%"]})
    out = cleaner.clean_values(df)
    assert out["Lab"].iloc[0] == 25.0


def test_em_dash_becomes_null_and_not_zero():
    """'not prompted' and
    'polled at zero per cent' are different claims about the world."""
    df = pd.DataFrame({"Lab": ["25%"], "PC": ["-"]})
    out = cleaner.clean_values(df)
    assert pd.isna(out["PC"].iloc[0])
    assert out["PC"].iloc[0] != 0


def test_footnote_markers_are_stripped_from_pollster_names():
    df = pd.DataFrame({"Pollster": ["YouGov[3]"]})
    assert cleaner.clean_values(df)["Pollster"].iloc[0] == "YouGov"


def test_sample_sizes_are_coerced_and_unusable_values_nulled():
    df = pd.DataFrame({"Sample size": ["2,101", "TBA", "-"]})
    out = cleaner.clean_values(df)
    assert out["Sample size"].iloc[0] == 2101
    assert pd.isna(out["Sample size"].iloc[1])
    assert pd.isna(out["Sample size"].iloc[2])


# pollster identity

@pytest.mark.parametrize("raw,expected", [
    ("Techne Archived 12 January 2025 at the Wayback Machine", "Techne"),
    ("Deltapoll[permanent dead link]", "Deltapoll"),
    ("Savanta ComRes", "Savanta"),
    ("Omnisis", "We Think"),
    ("Kantar Public", "Verian"),
    ("Find Out Now/ Electoral Calculus", "Find Out Now/Electoral Calculus"),
    ("FindOutNow/Electoral Calculus", "Find Out Now/Electoral Calculus"),
])
def test_pollster_identity_is_normalised(raw, expected):
    """fragmented identity risks producing a working system w quietly problematic
    output: one pollster presented as several, for example, w each holding part of a record."""
    name, _method, _client = cleaner.normalise_pollster(raw)
    assert name == expected


def test_survey_method_is_lifted_out_of_the_pollster_name():
    """YouGov and YouGov (MRP) are one organisation using 2 instruments,
    the weighting layer uses this distinction."""
    name, method, _client = cleaner.normalise_pollster("YouGov (MRP)")
    assert name == "YouGov"
    assert method == "MRP"


def test_standard_polls_are_marked_as_such():
    _name, method, _client = cleaner.normalise_pollster("Opinium")
    assert method == "standard"


# dates

def _dates(raw, year="2026"):
    df = pd.DataFrame({"Date(s) conducted": [raw], "source_year": [year]})
    out = cleaner.parse_dates(df)
    return out["fieldwork_start"].iloc[0], out["fieldwork_end"].iloc[0]


def test_range_within_one_month_borrows_the_end_month():
    start, end = _dates("7–10 Jul")
    assert (start.month, start.day) == (7, 7)
    assert (end.month, end.day) == (7, 10)


def test_range_across_a_month_boundary_keeps_its_own_months():
    start, end = _dates("28 Jun – 1 Jul")
    assert (start.month, start.day) == (6, 28)
    assert (end.month, end.day) == (7, 1)


def test_single_day_fieldwork_yields_equal_bounds():
    start, end = _dates("3 Jun")
    assert start == end


def test_year_boundary_is_corrected_by_impossible_ordering():
    """a range straddling a year boundary receives the section's year for both dates,
    so the start resolves AFTER the end to signal it for resolution."""
    start, end = _dates("30 Dec – 2 Jan", year="2026")
    assert start.year == 2025 and start.month == 12
    assert end.year == 2026 and end.month == 1
    assert start < end


# row routing

def test_event_rows_are_routed_out_of_the_polls():
    """event annotations carry no numeric party values bc they record
    no measurement."""
    df = pd.DataFrame({
        "Pollster": ["YouGov", "Partygate scandal begins"],
        "fieldwork_end": pd.to_datetime(["2026-07-10", "2021-12-07"]),
        "Lab": [25.0, np.nan], "Con": [24.0, np.nan],
    })
    polls, events = cleaner.split_events_from_polls(df)
    assert len(polls) == 1 and polls["Pollster"].iloc[0] == "YouGov"
    assert len(events) == 1
    assert events["description"].iloc[0] == "Partygate scandal begins"


def test_election_results_are_separated_from_polls(page_with_result):
    """a result is shaped like a poll but is not one, and must not be
    weighted alongside polls attempting to predict it."""
    tables = cleaner.select_national_tables(page_with_result, ["2024"])
    df = cleaner.canonicalise_columns(cleaner.flatten_headers(tables[0]))
    df = cleaner.parse_dates(cleaner.clean_values(df))
    polls, results = cleaner.extract_election_results(df)
    assert len(results) == 1
    assert "general election" in results["Pollster"].iloc[0].lower()
    assert not polls["Pollster"].str.contains("general election",
                                              case=False).any()


def test_election_result_dates_are_recovered(page_with_result):
    """result rows carry a full date, which the year-appending poll parser
    cannot process; they are parsed separately so the most significant date
    on the timeline is not lost."""
    tables = cleaner.select_national_tables(page_with_result, ["2024"])
    df = cleaner.canonicalise_columns(cleaner.flatten_headers(tables[0]))
    df = cleaner.parse_dates(cleaner.clean_values(df))
    _polls, results = cleaner.extract_election_results(df)
    assert results["fieldwork_end"].iloc[0] == pd.Timestamp("2024-07-04")


# multi-source concerns

def test_polls_appearing_on_two_source_pages_are_de_duplicated():
    """source pages overlap at an election boundary; identity is the house,
    the fieldwork dates and the sample size."""
    df = pd.DataFrame({
        "Pollster": ["YouGov", "YouGov", "Opinium"],
        "fieldwork_start": pd.to_datetime(["2024-07-01"] * 2 + ["2024-07-02"]),
        "fieldwork_end": pd.to_datetime(["2024-07-03"] * 2 + ["2024-07-04"]),
        "Sample size": [2000, 2000, 1500],
    })
    assert len(cleaner.dedupe_polls(df)) == 2


# e2e through cleaning pipeline

def test_pipeline_produces_polls_and_events_from_a_page(cached_page):
    polls, events = cleaner.clean_one_source(
        cached_page, ["2026", "2025"], "fixture")
    assert len(polls) == 6
    assert len(events) == 1
    assert polls["source"].unique().tolist() == ["fixture"]


def test_pipeline_preserves_not_prompted_values_end_to_end(cached_page):
    """the decision must survive every stage
    a null here later becomes an absent database row."""
    polls, _events = cleaner.clean_one_source(
        cached_page, ["2026", "2025"], "fixture")
    deltapoll = polls[polls["Pollster"] == "Deltapoll"]
    assert pd.isna(deltapoll["Ref"].iloc[0])
    assert pd.isna(deltapoll["PC"].iloc[0])
    assert deltapoll["Lab"].iloc[0] == 25.0


def test_pipeline_leaves_no_unparsed_dates(cached_page):
    polls, _events = cleaner.clean_one_source(
        cached_page, ["2026", "2025"], "fixture")
    assert polls["fieldwork_end"].isna().sum() == 0