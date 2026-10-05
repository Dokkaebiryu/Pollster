from pathlib import Path
import sys
import time
import requests

# one entry per source page. the key is used by cleaner.SOURCES to pair a downloaded page w
# the year-headings that should be read from  it
SOURCES = {
    "next": (
        "https://en.wikipedia.org/wiki/"
        "Opinion_polling_for_the_next_United_Kingdom_general_election",
        Path("data_cache/polling_page.html"),
    ),
    "2024": (
        "https://en.wikipedia.org/wiki/"
        "Opinion_polling_for_the_2024_United_Kingdom_general_election",
        Path("data_cache/polling_2024.html"),
    ),
    "2019": (
        "https://en.wikipedia.org/wiki/"
        "Opinion_polling_for_the_2019_United_Kingdom_general_election",
        Path("data_cache/polling_2019.html"),
    ),
}

HEADERS = {
    "User-Agent": "PollsterFYP/0.1 (University Final Year Project (KU); contact: rookssalvation@gmail.com)"
}
# courtesy pause between requests so several pages are not hammered at once.
DELAY_SECONDS = 1.0


def save_page(url: str, cache_path: Path) -> None:
    cache_path.parent.mkdir(exist_ok=True)   # create data_cache/ if absent
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    size_kb = cache_path.stat().st_size / 1024
    print(f"Saved {cache_path}  ({size_kb:.0f} KB)")

def save_all(keys=None) -> None:
    """download every configured source (or just the named ones)."""
    keys = keys or list(SOURCES)
    for i, key in enumerate(keys):
        if key not in SOURCES:
            print(f"  ! unknown source '{key}' - known: {list(SOURCES)}")
            continue
        url, cache_path = SOURCES[key]
        print(f"[{key}] {url}")
        save_page(url, cache_path)
        if i < len(keys) - 1:
            time.sleep(DELAY_SECONDS)

if __name__ == "__main__":
    # `python fetch_page.py` fetches everything;
    # `python fetch_page.py 2024` fetches just that page.
    save_all(sys.argv[1:] or None)