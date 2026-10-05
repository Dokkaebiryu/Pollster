"""
data adapter

fetches wikipedia polling tables (in this first instance - opinion polling for the next UKGE) 
reads them into pandas, and prints a summary so that the raw data can be read/seen before 
building anything on top of it.

this layer exists only to inspect the raw material. 
"""
from io import StringIO
import requests
import pandas as pd

# single source for the MVP's main topic: GB/UK Westminster voting intention.
SOURCE_URL = (
    "https://en.wikipedia.org/wiki/"
    "Opinion_polling_for_the_next_United_Kingdom_general_election"
)

# identification necessary in line w wikimedia foundation's user-agent policy (failure to properly identify can == ip address block/rate limitation)
HEADERS = {
    "User-Agent": "PollsterFYP/0.1 (University Final Year Project (KU); contact: rookssalvation@gmail.com)"
}


def fetch_html(url: str) -> str:
    """downloading page HTML, failing loudly if anything goes wrong."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()   # raises error on a non-200 response (responses outside of standard 200 OK success message)
    return response.text


def read_polling_tables(html: str) -> list[pd.DataFrame]:
    """returning every table on the page that mentions 'pollster'.

    matching on this word keeps the actual polling tables and skips
    navigation boxes and infoboxes, which won't contain it.
    """
    tables = pd.read_html(StringIO(html), match="Pollster")
    if not tables:
        raise ValueError(
            "no polling tables found - the page structure may have changed."
        )
    return tables


def main() -> None:
    print(f"Fetching: {SOURCE_URL}\n")
    html = fetch_html(SOURCE_URL)

    tables = read_polling_tables(html)
    print(f"Found {len(tables)} candidate polling table(s).\n")

    # one-line summary of each table to see overall shape.
    for i, table in enumerate(tables):
        print(f"--- Table {i}: {table.shape[0]} rows x {table.shape[1]} columns ---")
        print("Columns:", list(table.columns))
        print()

    # zoom on latest table.
    print("=" * 60)
    print("First few rows of Table 0:\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(tables[0].head(8))


if __name__ == "__main__":
    main()