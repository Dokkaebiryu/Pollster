from datetime import datetime
from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session

import cleaner
from database import (Base, get_engine, Topic, Pollster, Subject, Poll,
                      PollResult, PoliticalEvent)

# provenance per source page. the cleaner tags every row with a `source`
# key; these maps turn that key back into the URL it came from and the
# cache file whose mtime records when it was retrieved. a poll from the
# 2024 page must not be recorded as having come from the current one.
SOURCE_URLS = {
    "next": ("https://en.wikipedia.org/wiki/"
             "Opinion_polling_for_the_next_United_Kingdom_general_election"),
    "2024": ("https://en.wikipedia.org/wiki/"
             "Opinion_polling_for_the_2024_United_Kingdom_general_election"),
    "2019": ("https://en.wikipedia.org/wiki/"
             "Opinion_polling_for_the_2019_United_Kingdom_general_election"),
}
SOURCE_CACHES = {
    "next": Path("data_cache/polling_page.html"),
    "2024": Path("data_cache/polling_2024.html"),
    "2019": Path("data_cache/polling_2019.html"),
}

# fallbacks for rows with no source tag (i.e. data cleaned by an older
# single-page pipeline).
DEFAULT_SOURCE = "next"
SOURCE_URL = SOURCE_URLS[DEFAULT_SOURCE]
CACHE_PATH = SOURCE_CACHES[DEFAULT_SOURCE]


# subjects this topic measures, with display metadata for later use.
# "aggregate" marks Others as a sum of minor parties, not a single party.
SUBJECTS = [
    ("Lab", "party", "#E4003B"), # official party colour according to brand guidelines
    ("Con", "party", "#0087DC"), # official party colour according to brand guidelines
    ("Ref", "party", "#12B6CF"), # widely used colour for Reform web & print graphics (alternatives include #17B9D1 (primary colour in non-web/print contexts) and #0A1628 (secondary colour))
    ("LD",  "party", "#FF6400"), # recommended party colour for digital & web use according to brand guidelines (alternatives include their distinct orange #FAA61A)
    ("Grn", "party", "#00A85A"), # official party colour according to brand guidelines (additional brand colours inc. #00643B (Fun Green) & #28AA1E (Strong Green))
    ("SNP", "party", "#FFF48B"), # standard representation for UK Parliamentary Infographics & Broadcast Graphics (official yellow is #FFF48B, #FFF00 for standard digital rendering. its official black is obviously unsuitable)
    ("PC",  "party", "#0AA77D"), # official party  colour according to brand guidelines (may need to be changed to reduce clash with greens at a later point)
    ("RB",  "party", "#051D3F"), # official party colour according to brand guidelines
    ("UKIP", "party", "#70147A"), # official UKIP purple
    ("Brx",  "party", "#12B6CF"), # Brexit Party teal (brand cyan)
    ("Chg",  "party", "#222221"), # Change UK / TIG near-black
    ("Others", "aggregate", "#888888"), #neutral colour choice
]
PARTY_COLUMNS = [name for name, _type, _colour in SUBJECTS]


def to_date(value):
    """pandas Timestamp/NaT -> date or None."""
    return None if pd.isna(value) else pd.Timestamp(value).date()

def none_if_nan(value):
    return None if pd.isna(value) else str(value)


def retrieved_at(source: str = DEFAULT_SOURCE) -> datetime:
    """When cached page was fetched (file modification time)."""
    path = SOURCE_CACHES.get(source, CACHE_PATH)
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now()

def source_of(row) -> str:
    """the source key a cleaned row carries, defaulting for older data."""
    value = row.get("source")
    return DEFAULT_SOURCE if pd.isna(value) else str(value)


def load(polls: pd.DataFrame, events: pd.DataFrame) -> None:
    engine = get_engine()
    # deterministic rebuild from source.
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    # one retrieval timestamp per source page.
    fetched_by_source = {key: retrieved_at(key) for key in SOURCE_CACHES}

    with Session(engine) as session:
        # seeds single MVP topic.
        topic = Topic(name="Westminster voting intention",
                      subject_type="party", value_type="share")
        session.add(topic)

        # seeds subjects (parties + Others aggregate).
        subject_by_name = {}
        for name, stype, colour in SUBJECTS:
            subj = Subject(name=name, type=stype, short_code=name, colour_hex=colour)
            session.add(subj)
            subject_by_name[name] = subj

        # pollsters: get-or-create, so each distinct pollster is stored once (& not duplicated).
        pollster_by_name = {}

        def get_pollster(name: str) -> Pollster:
            if name not in pollster_by_name:
                p = Pollster(name=name, bpc_member=None)
                session.add(p)
                pollster_by_name[name] = p
            return pollster_by_name[name]

        # polls & their results.
        n_results = 0
        for _, row in polls.iterrows():
            pollster = get_pollster(str(row["Pollster"]))
            sample = pd.to_numeric(row.get("Sample size"), errors="coerce")

            src_key = source_of(row)

            poll = Poll(
                topic=topic,
                pollster=pollster,
                fieldwork_start=to_date(row.get("fieldwork_start")),
                fieldwork_end=to_date(row.get("fieldwork_end")),
                sample_size=None if pd.isna(sample) else int(sample),
                area=none_if_nan(row.get("Area")),
                client=none_if_nan(row.get("Client")),
                # "MRP"/"SRP"/"standard", lifted out of the pollster name by
                # the cleaner; the weighting layer uses this to stop modelled
                # polls dominating on their very large nominal sample sizes.
                methodology_type=none_if_nan(row.get("method")),
                source_url=SOURCE_URLS.get(src_key, SOURCE_URL),
                retrieved_at=fetched_by_source.get(src_key,
                                                   retrieved_at(src_key)),
            )
            session.add(poll)

            # wide -> long: one result per prompted subject.
            # non-prompted parties (NaN) simply produces no row.
            for name in PARTY_COLUMNS:
                value = row.get(name)
                if pd.notna(value):
                    session.add(PollResult(
                        poll=poll, subject=subject_by_name[name],
                        value=float(value)))
                    n_results += 1

        # political events extracted during cleaning.
        for _, ev in events.iterrows():
            session.add(PoliticalEvent(
                event_date=to_date(ev.get("fieldwork_end")),
                title=str(ev["description"]),
                source_url=SOURCE_URLS.get(source_of(ev), SOURCE_URL),
            ))

        session.commit()
        print(f"Loaded: {len(pollster_by_name)} pollsters, "
              f"{len(SUBJECTS)} subjects, {len(polls)} polls, "
              f"{n_results} poll results, {len(events)} events.")
        if "source" in polls.columns:
            print("  by source: " + ", ".join(
                f"{k}={v}" for k, v in polls["source"].value_counts().items()))
        if "method" in polls.columns:
            print("  by method: " + ", ".join(
                f"{k}={v}" for k, v in polls["method"].value_counts().items()))


def main() -> None:
    print("Building clean data...")
    polls, events = cleaner.get_clean_data()
    print("Writing to database...")
    load(polls, events)


if __name__ == "__main__":
    main()