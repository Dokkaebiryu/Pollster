import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

import database
import loader
from database import (Base, Poll, PollResult, Pollster, Subject, Topic,
                      PoliticalEvent)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """a throwaway database per test, isolated from the project's."""
    url = f"sqlite:///{tmp_path / 'test.sqlite'}"
    monkeypatch.setattr(database, "DB_URL", url)
    monkeypatch.setattr(loader, "get_engine",
                        lambda echo=False: database.create_engine(url))
    return database.create_engine(url)


@pytest.fixture
def polls_frame():
    """2 polls. the 2nd was not prompted for Reform or Plaid Cymru
    and must therefore produce fewer results than the
    first, not two zeroes."""
    return pd.DataFrame([
        {"Pollster": "YouGov", "Client": "The Times", "Area": "GB",
         "Sample size": 2000, "method": "standard", "source": "next",
         "fieldwork_start": pd.Timestamp("2026-07-01"),
         "fieldwork_end": pd.Timestamp("2026-07-03"),
         "Lab": 25.0, "Con": 24.0, "Ref": 23.0, "LD": 12.0, "Grn": 8.0,
         "SNP": 3.0, "PC": 1.0, "RB": 2.0, "Others": 2.0},
        {"Pollster": "Opinium", "Client": None, "Area": "UK",
         "Sample size": 1800, "method": "MRP", "source": "2024",
         "fieldwork_start": pd.Timestamp("2026-07-04"),
         "fieldwork_end": pd.Timestamp("2026-07-05"),
         "Lab": 26.0, "Con": 23.0, "Ref": float("nan"), "LD": 13.0,
         "Grn": 9.0, "SNP": 3.0, "PC": float("nan"), "RB": 2.0,
         "Others": 2.0},
    ])


@pytest.fixture
def events_frame():
    return pd.DataFrame([
        {"fieldwork_end": pd.Timestamp("2024-07-04"),
         "description": "2024 general election", "source": "2024"},
    ])


@pytest.fixture
def loaded(engine, polls_frame, events_frame):
    loader.load(polls_frame, events_frame)
    return engine


# central claim: absence is represented by absence

def test_a_party_not_prompted_produces_no_result_row(loaded):
    """the decision on which the system's honesty is based. a zero here would
    assert that a party had no support when no measurement was taken."""
    with Session(loaded) as s:
        opinium = s.scalar(select(Poll).join(Pollster)
                           .where(Pollster.name == "Opinium"))
        measured = {r.subject.name for r in opinium.results}
    assert "Ref" not in measured
    assert "PC" not in measured
    assert "Lab" in measured


def test_no_result_row_anywhere_holds_a_fabricated_zero(loaded):
    with Session(loaded) as s:
        values = s.scalars(select(PollResult.value)).all()
    assert all(v > 0 for v in values)


def test_result_counts_reflect_what_was_actually_prompted(loaded):
    """9 subjects across two polls would give 18 rows if every party
    were recorded for every poll; 2 were not prompted, so 16."""
    with Session(loaded) as s:
        assert len(s.scalars(select(PollResult)).all()) == 16


# normalisation of entities

def test_each_pollster_is_stored_once(engine, polls_frame, events_frame):
    """get-or-create: the same house appearing in many polls must not
    produce many pollster rows."""
    repeated = pd.concat([polls_frame, polls_frame], ignore_index=True)
    loader.load(repeated, events_frame)
    with Session(engine) as s:
        names = s.scalars(select(Pollster.name)).all()
    assert sorted(names) == ["Opinium", "YouGov"]


def test_subjects_are_seeded_with_display_metadata(loaded):
    with Session(loaded) as s:
        subjects = s.scalars(select(Subject)).all()
    assert len(subjects) == 12
    assert all(sub.colour_hex for sub in subjects)
    assert {sub.type for sub in subjects} == {"party", "aggregate"}


def test_a_single_topic_is_seeded_with_its_value_semantics(loaded):
    with Session(loaded) as s:
        topic = s.scalar(select(Topic))
    assert topic.subject_type == "party"
    assert topic.value_type == "share"


# provenance

def test_every_poll_records_where_it_came_from(loaded):
    with Session(loaded) as s:
        polls = s.scalars(select(Poll)).all()
    assert all(p.source_url for p in polls)
    assert all(p.retrieved_at for p in polls)


def test_provenance_is_recorded_per_source_not_globally(loaded):
    """a poll from the 2024 election page must not be recorded as having come
    from the current one."""
    with Session(loaded) as s:
        by_pollster = {p.pollster.name: p.source_url
                       for p in s.scalars(select(Poll)).all()}
    assert by_pollster["YouGov"] != by_pollster["Opinium"]
    assert "2024" in by_pollster["Opinium"]


def test_survey_method_is_stored_against_the_poll(loaded):
    """lifted out of the pollster name during cleaning; the weighting layer
    depends on being able to tell a modelled poll from a regularly sampled one."""
    with Session(loaded) as s:
        methods = {p.pollster.name: p.methodology_type
                   for p in s.scalars(select(Poll)).all()}
    assert methods["Opinium"] == "MRP"
    assert methods["YouGov"] == "standard"


def test_poll_metadata_survives_the_load(loaded):
    with Session(loaded) as s:
        yougov = s.scalar(select(Poll).join(Pollster)
                          .where(Pollster.name == "YouGov"))
    assert yougov.sample_size == 2000
    assert yougov.area == "GB"
    assert yougov.client == "The Times"
    assert yougov.fieldwork_start.isoformat() == "2026-07-01"
    assert yougov.fieldwork_end.isoformat() == "2026-07-03"


def test_events_are_stored_separately_from_polls(loaded):
    with Session(loaded) as s:
        events = s.scalars(select(PoliticalEvent)).all()
        poll_count = len(s.scalars(select(Poll)).all())
    assert len(events) == 1
    assert events[0].title == "2024 general election"
    assert poll_count == 2


# rebuild semantics

def test_loading_twice_rebuilds_rather_than_accumulating(engine, polls_frame,
                                                          events_frame):
    """the load is deliberately destructive: the database holds nothing that
    cannot be re-derived, and a rebuild guarantees its contents match the
    current pipeline rather than a mixture of versions."""
    loader.load(polls_frame, events_frame)
    loader.load(polls_frame, events_frame)
    with Session(engine) as s:
        assert len(s.scalars(select(Poll)).all()) == 2
        assert len(s.scalars(select(Pollster)).all()) == 2