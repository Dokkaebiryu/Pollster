import re
from io import StringIO
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup

CACHE_PATH = Path("data_cache/polling_page.html")

# one entry per source page. each page carries the same table SHAPE but a
# different span of years, so the wanted headings differ per source. adding
# an older election page later is a matter of adding a row here.
SOURCES = {
    "next": {
        "cache": Path("data_cache/polling_page.html"),
        "years": ["2026", "2025", "2024"],
    },
    "2024": {
        "cache": Path("data_cache/polling_2024.html"),
        "years": ["2024", "2023", "2022", "2021", "2020"],
    },
    "2019": {
        "cache": Path("data_cache/polling_2019.html"),
        "years": ["2019", "2018", "2017"],
    },
}

# section headings (on the page) whose tables ARE wanted.
# (for now, these are the national voting-intention tables, split by year from 2024 - 2026)
WANTED_YEARS = SOURCES["next"]["years"]


def load_html(path: Path = CACHE_PATH) -> str:
    """read a cached page from disk (no network call)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run fetch_page.py first."
        )
    return path.read_text(encoding="utf-8")


def select_national_tables(html: str, wanted_years=None) -> list[pd.DataFrame]:
    """returns only the national year-tables, chosen by their heading.

    the page is walked heading-by-heading. when a heading whose
    text is one of the WANTED_YEARS is reached, the first table that follows it.
    is taken.
    this avoids selecting hypothetical-scenario tables, which share an
    identical column structure with the real national tables and so
    cannot be told apart by their columns alone.
    """
    wanted_years = wanted_years or WANTED_YEARS
    soup = BeautifulSoup(html, "html.parser")
    selected = []

    for heading in soup.find_all(["h2", "h3", "h4"]):
        title = heading.get_text().strip()
        if title in wanted_years:
            # finds the first <table> appearing after this heading.
            table = heading.find_next("table")
            if table is not None:
                # re-parses this one table's HTML into a dataframe.
                df = pd.read_html(StringIO(str(table)))[0]
                df["source_year"] = title      # remember which year it came from
                selected.append(df)
                print(f"  matched heading '{title}': {df.shape[0]} rows")

    if not selected:
        raise ValueError(
            "No national tables matched. The page headings may have changed."
        )
    return selected


def flatten_headers(df: pd.DataFrame) -> pd.DataFrame:
    """collapse the two-level column headers to a single level.

    pandas read the table's two header rows as pairs, e.g.
    ('Lab', 'Unnamed: 5_level_1'). the real name is always in the first
    element; the second is superfluous. so, the first element of each
    pair is kept and the other discarded. 
    """
    df = df.copy()
    if not isinstance(df.columns, pd.MultiIndex):
        # some (older) pages use a single header row already.
        df.columns = [str(c).strip() for c in df.columns]
        return df
    flat = []
    for parts in df.columns:
        # the real name is always the first element; the rest are either
        # repeats or pandas' "Unnamed: n_level_n" filler.
        top = str(parts[0]).strip()
        flat.append(top)
    df.columns = flat
    return df

# column headings drift between source pages: the current page writes
# "Date(s) conducted" while the 2024 page writes "Dates conducted". rather
# than hard-code every variant, each heading is matched loosely and renamed
# to one canonical name that the rest of the pipeline can rely on.
CANONICAL_COLUMNS = {
    "Date(s) conducted": [r"date.*conduct"],
    "Pollster": [r"^pollster"],
    "Client": [r"^client"],
    "Area": [r"^area"],
    "Sample size": [r"sample\s*size"],
}


def canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """rename loosely-matched headings to their canonical form.

    matching is case-insensitive and substring-based, so wording changes
    between pages (e.g. "Dates conducted" vs "Date(s) conducted") do not
    break the pipeline. the FIRST unclaimed match wins, so a page with two
    similar headings does not have both collapsed onto one name.
    """
    df = df.copy()
    renames, claimed = {}, set()
    for canonical, patterns in CANONICAL_COLUMNS.items():
        for col in df.columns:
            if col in claimed or col == canonical:
                if col == canonical:
                    claimed.add(col)
                continue
            text = str(col).strip().lower()
            if any(re.search(p, text, flags=re.I) for p in patterns):
                renames[col] = canonical
                claimed.add(col)
                break
    if renames:
        df = df.rename(columns=renames)
    return df

# pollster names drift across the source pages in three ways, all of which
# would otherwise fragment one polling house into several `pollster` rows:
#   1. wikipedia link-rot suffixes ("Archived ... at the Wayback Machine",
#      "[dead link]") get concatenated into the cell text.
#   2. survey method is appended in brackets - "YouGov (MRP)". MRP is a
#      different instrument, so it is split OUT into its own column rather
#      than being allowed to masquerade as a separate pollster.
#   3. houses rebrand or are acquired. wikipedia states these particular
#      changes did not entail changes of methodology, so they are merged.
POLLSTER_ALIASES = {
    "ComRes": "Savanta",
    "Savanta ComRes": "Savanta",
    "Omnisis": "We Think",
    "Kantar Public": "Verian",
    "Kantar": "Verian",
    "Panelbase": "Norstat",
    "ICM": "ICM Unlimited",
    "FindOutNow": "Find Out Now",
    "Ipsos MORI": "Ipsos",
    "Number Cruncher": "Number Cruncher Politics",
    "FindOutNow": "Find Out Now",
    "FindOutNow/Electoral Calculus": "Find Out Now/Electoral Calculus",
    "Lord Ashcroft Polls": "Lord Ashcroft",
    "Ipsos MORI": "Ipsos",
    "Number Cruncher": "Number Cruncher Politics",
}

# joint products where the "/" is part of the pollster's own identity, not a
# pollster/client separator. these must NOT be split on the slash.
JOINT_POLLSTERS = {
    "find out now/electoral calculus",
    "findoutnow/electoral calculus",
}

# bracketed method markers appearing after a pollster's name.
METHOD_MARKERS = ["MRP", "SRP"]

def normalise_pollster(raw: str) -> tuple[str, str]:
    """returns (clean_pollster_name, method).

    method is "MRP"/"SRP" where the source flagged one, else "standard".
    the sample size is left untouched: it is real provenance. what changes
    is that the WEIGHTING layer can now see which polls are modelled
    estimates rather than straight samples (see trend.py).
    """
    name = str(raw)
    # 1. strip link-rot debris.
    name = re.sub(r"Archived.*?(Wayback Machine|$)", "", name)
    name = re.sub(r"\[[^\]]*\]", "", name)          # [dead link], [2], ...
    name = re.sub(r"\s+", " ", name).strip(" .,;·")

    # 2. lift any bracketed method marker out of the name.
    method = "standard"
    for marker in METHOD_MARKERS:
        pattern = rf"\(\s*{marker}\s*\)"
        if re.search(pattern, name, flags=re.I):
            method = marker
            name = re.sub(pattern, "", name, flags=re.I)
    # Removing "(MRP)" leaves the space that preceded it, so the collapse
    # and strip must follow the removal, not precede it. Without this,
    # "YouGov (MRP)" normalises to "YouGov " and is stored as a pollster
    # distinct from "YouGov" - the exact fragmentation this routine exists
    # to prevent, and invisible in the interface.
    name = re.sub(r"\s+", " ", name).strip()
    # joint products are written inconsistently across the source pages
    # ("Find Out Now/ Electoral Calculus" vs "FindOutNow/Electoral
    # Calculus"), so spacing around the slash is normalised before aliasing.
    name = re.sub(r"\s*/\s*", "/", name)

    # Older pages write the commissioning client into the pollster cell as
    # "Pollster/Client" (e.g. "ComRes/Daily Express", "YouGov/The Times").
    # The pollster is the segment before the first slash. Genuine joint
    # products, where the slash is part of the pollster's identity, are
    # exempted — splitting them would fragment one pollster into two.
    embedded_client = None
    if "/" in name and name.lower() not in JOINT_POLLSTERS:
        name, embedded_client = (part.strip() for part in name.split("/", 1))

    # 3. merge rebrands onto their current identity.
    return POLLSTER_ALIASES.get(name, name), method, embedded_client

# columns that hold party's vote share (as text like "19%" or "-").
PARTY_COLUMNS = ["Lab", "Con", "Ref", "LD", "Grn", "SNP", "PC", "RB",
                 "UKIP", "Brx", "Chg", "Others"]

def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """turn text percentages into numbers, strip footnotes.

    2 distinct conversions:
      - party columns: "19%" -> 19.0 ; "-" (did not stand) -> NaN (missing).
        "-" is NOT turned into 0: "did not stand" and "polled
        at 0%" are different facts, and conflating them would 
        risk data misapprehension
      - pollster/client: strip trailing footnote markers like "[2]".
    """
    df = df.copy()

    df = df.copy()

    # Older source pages change their party lineup mid-period and so use more
    # than one table layout. Concatenating those tables leaves pandas with
    # DUPLICATE columns of the same name (e.g. the 2019 page carries two
    # 'Grn'/'PC'/'UKIP' blocks, one per layout), each row populated in only
    # one. Selecting such a name returns a DataFrame, not a Series. These are
    # coalesced — first non-null across the duplicates — rather than dropped,
    # so no party's data is silently lost to a layout change.
    if df.columns.duplicated().any():
        merged = {}
        for name in df.columns[df.columns.duplicated()].unique():
            block = df.loc[:, df.columns == name]
            merged[name] = block.bfill(axis=1).iloc[:, 0]
        df = df.loc[:, ~df.columns.duplicated(keep=False)]
        for name, series in merged.items():
            df[name] = series

    # strip footnote markers from text columns.
    for col in ["Pollster", "Client"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()

    # normalise pollster identity and lift the survey method into its own
    # column (so "YouGov" and "YouGov (MRP)" are one house, two methods).
    if "Pollster" in df.columns:
        normalised = df["Pollster"].apply(normalise_pollster)
        df["Pollster"] = [n for n, _m, _c in normalised]
        df["method"] = [m for _n, m, _c in normalised]
        embedded = [c for _n, _m, c in normalised]
        if "Client" in df.columns:
            # keep the page's own client column where it exists; fall back to
            # the client embedded in "Pollster/Client" only where it's blank.
            df["Client"] = [existing if pd.notna(existing) else emb
                            for existing, emb in zip(df["Client"], embedded)]
        else:
            df["Client"] = embedded

    # sample size arrives as "2,101" / "TBA" / "?" / "–"; anything that is
    # not a number becomes NaN so the weighting layer can exclude it.
    if "Sample size" in df.columns:
        df["Sample size"] = pd.to_numeric(
            df["Sample size"].astype(str)
              .str.replace(",", "", regex=False)
              .str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce")

    # convert party % strings to numbers; em-dash -> missing.
    for col in PARTY_COLUMNS:
        if col in df.columns:
            cleaned = (
                df[col].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace("-", "", regex=False)   # em-dash -> empty
                .str.strip()
            )
            # empty string (was a dash) becomes NaN, not 0.
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """parse the fieldwork range into fieldwork_start and fieldwork_end.

    the source gives year-less ranges like '7–10 Jul' or, across a month
    boundary, '28 Jun–1 Jul'. the END always carries its own month; the
    START sometimes does and sometimes borrows the end's month. the end is
    parsed first, then the start, giving it the end's month/yr only when
    it lacks a month of its own. if the resulting start comes out after
    the end, the range crossed a year boundary, and so the start must
    belong to a previous year which should be corrected for.
    """
    df = df.copy()

    def split_range(raw: str) -> tuple[str, str]:
        parts = str(raw).split("–")            # dash
        if len(parts) == 1:
            return parts[0].strip(), parts[0].strip()   # single day
        return parts[0].strip(), parts[-1].strip()

    starts_end, ends, years = [], [], df["source_year"].astype(str)

    for raw in df["Date(s) conducted"]:
        start_raw, end_raw = split_range(raw)
        # if the start has no letters, it's just a day number and borrows
        # the end's month (e.g. '7' in '7–10 Jul').
        start_has_month = any(ch.isalpha() for ch in start_raw)
        starts_end.append((start_raw, end_raw, start_has_month))

    end_strs = [f"{e} {y}" for (_, e, _), y in zip(starts_end, years)]
    start_strs = [
        f"{s} {y}" if has_month else f"{s} {e.split()[-1]} {y}"
        for (s, e, has_month), y in zip(starts_end, years)
    ]

    df["fieldwork_end"] = pd.to_datetime(end_strs, format="%d %b %Y", errors="coerce")
    df["fieldwork_start"] = pd.to_datetime(start_strs, format="%d %b %Y", errors="coerce")

    # if start ends up after end, the range must have crossed a year boundary
    # (e.g. 30 Dec–2 Jan): the start is then pulled back by a year.
    crossed = df["fieldwork_start"] > df["fieldwork_end"]
    df.loc[crossed, "fieldwork_start"] -= pd.DateOffset(years=1)

    return df

def drop_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """remove fully-empty separator rows."""
    key_cols = ["Date(s) conducted", "Pollster"] + PARTY_COLUMNS
    present = [c for c in key_cols if c in df.columns]
    return df.dropna(subset=present, how="all").reset_index(drop=True)


def split_events_from_polls(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """separate embedded event-caption rows from 'true' poll rows.

    the original tables weave in political-event annotations (e.g. a leader's
    election) among the polls. such rows lack numeric party values, with their
    'pollster' cell holding a sentence rather than a pollster name. by the absence 
    of any numeric party values, these can be identified and routed into an events
    frame with the polls returned separately (and cleaned of them). 
    """
    party_present = [c for c in PARTY_COLUMNS if c in df.columns]
    has_no_numbers = df[party_present].isna().all(axis=1)

    events = df[has_no_numbers].copy()
    polls = df[~has_no_numbers].copy().reset_index(drop=True)

    # shapes events into something usable later: date + description.
    events = events[["fieldwork_end", "Pollster"]].rename(
        columns={"Pollster": "description"}
    )
    # a event row has a parsed date AND a real sentence; blank
    # separator rows survive as the string "nan" once cast to str above.
    events = events.dropna(subset=["description", "fieldwork_end"])
    events = events[events["description"].astype(str).str.strip()
                    .str.lower().ne("nan")]
    events = events.reset_index(drop=True)

    return polls, events

# rows whose "pollster" cell names an election result rather than a house.
ELECTION_ROW_PATTERN = r"general election"


def extract_election_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """split out general-election RESULT rows from the poll rows.

    the source tables carry the actual election result as a row shaped like
    a poll (possessing party %s), but it is not a poll and must not be
    weighted as one. it is, however, the most useful anchor on the timeline,
    so it is returned separately to be stored as a political event (and,
    later, as a true baseline for 'change since the election').
    """
    if "Pollster" not in df.columns:
        return df, df.iloc[0:0].copy()
    is_result = df["Pollster"].astype(str).str.contains(
        ELECTION_ROW_PATTERN, case=False, na=False)
    results = df[is_result].copy().reset_index(drop=True)
    polls = df[~is_result].copy().reset_index(drop=True)

    # election rows already carry a FULL date ("4 Jul 2024"), so the normal
    # year-appending parser produces "4 Jul 2024 2024" and fails. They are
    # parsed directly here instead.
    if len(results) and "Date(s) conducted" in results.columns:
        parsed = pd.to_datetime(results["Date(s) conducted"].astype(str).str.strip(),
                                dayfirst=True, errors="coerce")
        results["fieldwork_end"] = results["fieldwork_end"].fillna(parsed)
        results["fieldwork_start"] = results["fieldwork_start"].fillna(parsed)
    return polls, results


def drop_non_poll_rows(df: pd.DataFrame) -> pd.DataFrame:
    """remove rows that aren't polls or events (e.g. general-election result line).

    these are dropped so that the poll set is uncorrupted, containing only polls.
    the election result may be brought back in as an anchor point at a later stage,
    but it is not a poll and hence should not be treated as one. 
    """
    return df.dropna(subset=["fieldwork_end"]).reset_index(drop=True)

def clean_one_source(path: Path, wanted_years, source_key: str):
    """cleans a SINGLE cached page, returning (polls, events).

    this is the old single-page pipeline, lifted out so it can be run once
    per source page and the results concatenated.
    """
    html = load_html(path)
    tables = select_national_tables(html, wanted_years)
    combined = pd.concat(tables, ignore_index=True)
    combined = flatten_headers(combined)
    combined = canonicalise_columns(combined)
    # clean_values must run EXACTLY once. It is not idempotent: the method
    # extraction removes the "(MRP)" marker it reads, so a second pass sees
    # an already-normalised name, finds no marker, and overwrites every
    # modelled poll's method with "standard".
    combined = clean_values(combined)
    combined = parse_dates(combined)
    combined = drop_blank_rows(combined)
    polls, events = split_events_from_polls(combined)
    polls, results = extract_election_results(polls)
    polls = drop_non_poll_rows(polls)

    # election results become dated events so they appear on the timeline.
    if len(results):
        as_events = results[["fieldwork_end", "Pollster"]].rename(
            columns={"Pollster": "description"})
        as_events = as_events.dropna(subset=["fieldwork_end"])
        events = pd.concat([events, as_events], ignore_index=True)
        print(f"  captured {len(as_events)} election result row(s) as events")

    polls["source"] = source_key          # provenance: page it came from
    events["source"] = source_key
    return polls, events

def dedupe_polls(df: pd.DataFrame) -> pd.DataFrame:
    """drop polls appearing on more than one source page.

    the pages overlap at an election boundary (the 'next election' page and
    the 2024 page both carry a 2024 section, either side of polling day), so
    a poll can legitimately appear twice. identity is the combination of
    house, fieldwork dates and sample size.
    """
    keys = [c for c in ["Pollster", "fieldwork_start", "fieldwork_end",
                        "Sample size"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  de-duplicated {dropped} poll(s) appearing on two pages")
    return df

def get_clean_data(sources=None):
    """runs full cleaning pipeline across ALL sources; returns (polls, events).

    (this is the reusable entry point other modules import; 
    main() just calls it and prints, so the pipeline is defined in one place.)
    a source whose cache file is missing is SKIPPED with a warning rather
    than raising, so the pipeline still runs while pages are being added.
    """
    sources = sources or SOURCES
    all_polls, all_events = [], []

    for key, cfg in sources.items():
        if not cfg["cache"].exists():
            print(f"  ! skipping source '{key}': {cfg['cache']} not found")
            continue
        print(f"  reading source '{key}' ({cfg['cache'].name})")
        polls, events = clean_one_source(cfg["cache"], cfg["years"], key)
        all_polls.append(polls)
        all_events.append(events)

    if not all_polls:
        raise FileNotFoundError("No cached source pages found - run fetch_page.py first.")

    polls = pd.concat(all_polls, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True)
    polls = dedupe_polls(polls)

    # the same event is often carried by more than one source page, and
    # election results appear twice per page (UK-wide and GB-only rows).
    before = len(events)
    events = events.drop_duplicates(
        subset=["fieldwork_end", "description"]).reset_index(drop=True)
    if before - len(events):
        print(f"  de-duplicated {before - len(events)} event(s)")
    return polls, events

def main() -> None:
    print("Loading cached page...")
    polls, events = get_clean_data()

    print(f"\nPolls: {polls.shape[0]} rows    Events extracted: {events.shape[0]} rows")

    print("\nFirst few POLL rows (cleaned):\n")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(polls[["fieldwork_start", "fieldwork_end", "Pollster", "Client",
                     "Area", "Lab", "Con", "Ref", "LD", "Grn"]].head(8))

    print("\nEvents pulled out of the tables:\n")
    print(events.to_string())

    print("\n--- health checks ---")
    print("Poll rows where date failed to parse:",
          polls["fieldwork_end"].isna().sum())
    print("Not-prompted counts per party (blank = pollster didn't offer it, or otherwise not applicable):")
    print(polls[PARTY_COLUMNS].isna().sum())


if __name__ == "__main__":
    main()