from datetime import date, datetime
from sqlalchemy import (String, Integer, Float, Boolean, Date, DateTime,
                        ForeignKey, create_engine, inspect)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship)

DB_URL = "sqlite:///pollster.sqlite"


class Base(DeclarativeBase):
    pass


class Topic(Base):
    """the thing being polled, for example Westminster voting intention.
    subject_type sets whether its subjects are parties or leaders;
    value_type sets what the numbers mean (vote share, net approval, etc.,)."""
    __tablename__ = "topic"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    subject_type: Mapped[str] = mapped_column(String)   # "party" | "leader"
    value_type: Mapped[str] = mapped_column(String)     # "share" | "net" | "pref"
    polls: Mapped[list["Poll"]] = relationship(back_populates="topic")


class Pollster(Base):
    """an organisation that conducts polls (YouGov, Opinium, etc.,)."""
    __tablename__ = "pollster"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    bpc_member: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    polls: Mapped[list["Poll"]] = relationship(back_populates="pollster")


class Subject(Base):
    """the party or leader that a poll measures in a given column. 
    each subject is linked to a topic, and each poll result links a subject to a poll."""
    __tablename__ = "subject"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    type: Mapped[str] = mapped_column(String)           # "party"|"leader"|"aggregate"
    short_code: Mapped[str | None] = mapped_column(String, nullable=True)
    colour_hex: Mapped[str | None] = mapped_column(String, nullable=True)
    results: Mapped[list["PollResult"]] = relationship(back_populates="subject")


class Poll(Base):
    """one poll: who ran it, when, on what topic, + provenance."""
    __tablename__ = "poll"
    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topic.id"))
    pollster_id: Mapped[int] = mapped_column(ForeignKey("pollster.id"))
    fieldwork_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    fieldwork_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    area: Mapped[str | None] = mapped_column(String, nullable=True)
    client: Mapped[str | None] = mapped_column(String, nullable=True)
    methodology_type: Mapped[str | None] = mapped_column(String, nullable=True)
    margin_of_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    topic: Mapped["Topic"] = relationship(back_populates="polls")
    pollster: Mapped["Pollster"] = relationship(back_populates="polls")
    results: Mapped[list["PollResult"]] = relationship(
        back_populates="poll", cascade="all, delete-orphan")


class PollResult(Base):
    """one party's/leader's value in one poll. absence of a row means the
    pollster did not prompt for that subject (or that results were marginal)."""
    __tablename__ = "poll_result"
    id: Mapped[int] = mapped_column(primary_key=True)
    poll_id: Mapped[int] = mapped_column(ForeignKey("poll.id"))
    subject_id: Mapped[int] = mapped_column(ForeignKey("subject.id"))
    value: Mapped[float] = mapped_column(Float)
    poll: Mapped["Poll"] = relationship(back_populates="results")
    subject: Mapped["Subject"] = relationship(back_populates="results")


class PoliticalEvent(Base):
    """a dated political event, for context timeline."""
    __tablename__ = "political_event"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    title: Mapped[str] = mapped_column(String)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)


def get_engine(echo: bool = False):
    return create_engine(DB_URL, echo=echo)


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    print("Created database: pollster.sqlite\n")
    for t in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns(t)]
        print(f"  {t} ({len(cols)} cols): {', '.join(cols)}")


if __name__ == "__main__":
    init_db()