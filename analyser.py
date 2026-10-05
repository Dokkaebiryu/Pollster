import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import get_engine, Topic, Poll, Pollster, Subject, PollResult, PoliticalEvent

DEFAULT_TOPIC = "Westminster voting intention"


def load_topic_polls(topic_name: str = DEFAULT_TOPIC) -> pd.DataFrame:
    """returns one row per poll for a topic, with a column per subject.

    subjects a pollster did not prompt for come back as NaN (there was no
    poll_result row for them), preserving the not-prompted distinction.
    """
    engine = get_engine()
    with Session(engine) as session:
        topic = session.scalar(select(Topic).where(Topic.name == topic_name))
        if topic is None:
            raise ValueError(f"Topic not found: {topic_name!r}")

        stmt = (
            select(
                Poll.id.label("poll_id"),
                Poll.fieldwork_start,
                Poll.fieldwork_end,
                Poll.sample_size,
                # carried through so the interface can show a poll's scope
                # and origin beside its numbers (FR4, FR7). provenance that
                # is stored but never surfaced does not discharge a
                # transparency requirement.
                Poll.area,
                Poll.client,
                Poll.methodology_type,
                Poll.source_url,
                Poll.retrieved_at,
                Pollster.name.label("pollster"),
                Subject.name.label("subject"),
                PollResult.value,
            )
            .join(Pollster, Poll.pollster_id == Pollster.id)
            .join(PollResult, PollResult.poll_id == Poll.id)
            .join(Subject, PollResult.subject_id == Subject.id)
            .where(Poll.topic_id == topic.id)
        )
        long_df = pd.read_sql(stmt, session.connection())

    # long -> wide: subjects become columns.
    # Poll metadata is separated from the pivot rather than carried in its
    # index. Two reasons: a wide index makes the pivot allocate the full
    # cartesian product of its levels, and nullable columns (a missing
    # client, say) would otherwise cause the whole poll to be dropped.
    meta_cols = ["poll_id", "fieldwork_start", "fieldwork_end", "sample_size",
                 "area", "client", "methodology_type", "source_url",
                 "retrieved_at", "pollster"]
    meta = long_df[meta_cols].drop_duplicates("poll_id")

    # long -> wide on the identifier alone; subjects become columns.
    wide = long_df.pivot_table(index="poll_id", columns="subject",
                               values="value").reset_index()
    wide = meta.merge(wide, on="poll_id", how="left")

    wide.columns.name = None
    return wide.sort_values("fieldwork_end").reset_index(drop=True)

def load_events() -> pd.DataFrame:
    """returns dated political events for context timeline."""
    engine = get_engine()
    with Session(engine) as session:
        rows = session.execute(
            select(PoliticalEvent.event_date, PoliticalEvent.title)
        ).all()
    df = pd.DataFrame(rows, columns=["event_date", "title"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    return (df.dropna(subset=["event_date"])
              .sort_values("event_date").reset_index(drop=True))

def main() -> None:
    df = load_topic_polls()
    print(f"Loaded {len(df)} polls x {df.shape[1]} columns")
    print("Columns:", list(df.columns))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.tail(5))


if __name__ == "__main__":
    main()