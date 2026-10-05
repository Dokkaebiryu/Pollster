"""
pollster single-command run

checks for wikipedia source changes from cached copy & re-fetches + rebuilds the db for any that have changed.
unchanged sources are served from the cache, and if the network is unreachable the last cached data is served rather than failing (nfr6)

  python run.py            # refresh only what changed, then serve
  python run.py --refresh  # force a full re-fetch + rebuild
  python run.py --offline  # skip all network checks, serve cached data

the source map is imported from fetch_page rather than being duplicated,
so adding/changing a source is a one-place edit (nfr9)
"""
from __future__ import annotations
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import requests

import fetch_page  # for its SOURCES map; importing it has no side effects

LOADER_SCRIPT = "loader.py"
USER_AGENT = "Pollster/1.0 (FYP project)"  # descriptive UA, per Wikimedia policy
PROBE_TIMEOUT = 10  # seconds
MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "pollster.sqlite"


def _page_title(url: str) -> str:
    """Extract the article title from a /wiki/<Title> URL."""
    return unquote(url.rsplit("/wiki/", 1)[1])


def source_changed(url: str, cache_path: Path) -> bool:
    """True if the live page was edited after we cached it (or no cache yet).

    Queries the MediaWiki API for the page's latest revision timestamp and
    compares it to the cache file's mtime. This is used instead of a
    conditional GET because Wikipedia's If-Modified-Since / 304 handling is
    driven by its cache layer and does not reliably reflect article edits.

    Any network/parse error is treated as 'unchanged' so we degrade to the
    cached data rather than failing outright (NFR6).
    """
    if not cache_path.exists():
        return True
    params = {
        "action": "query", "prop": "revisions", "titles": _page_title(url),
        "rvprop": "timestamp", "rvlimit": 1, "format": "json",
        "formatversion": 2,
    }
    try:
        resp = requests.get(MEDIAWIKI_API, params=params,
                            headers={"User-Agent": USER_AGENT},
                            timeout=PROBE_TIMEOUT)
        resp.raise_for_status()
        ts = resp.json()["query"]["pages"][0]["revisions"][0]["timestamp"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"  ! could not check revision ({exc}); using cached copy")
        return False
    last_edit = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    cached_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    return last_edit > cached_at


def run_step(args: list[str]) -> None:
    """Run a pipeline script with the current interpreter; fail loudly."""
    printable = " ".join(args)
    print(f"-> {printable}")
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"Step failed: {printable} (exit {result.returncode})")


def main() -> None:
    flags = sys.argv[1:]
    refresh = "--refresh" in flags
    offline = "--offline" in flags
    changed = []

    for key, (url, cache_rel) in fetch_page.SOURCES.items():
        cache_path = ROOT / cache_rel  # resolve fetch_page's relative path
        if offline:
            if not cache_path.exists():
                sys.exit(f"--offline but no cache for [{key}] at {cache_rel}")
            print(f"[{key}] offline -- keeping cache")
            continue
        if refresh or source_changed(url, cache_path):
            print(f"[{key}] change detected -- fetching")
            run_step(["fetch_page.py", key])
            changed.append(key)
        else:
            print(f"[{key}] unchanged -- keeping cache")

    if refresh or changed or not DB_PATH.exists():
        why = "forced" if refresh else ("sources changed" if changed else "no database")
        print(f"rebuilding database ({why})")
        run_step([LOADER_SCRIPT])
    else:
        print("database current -- skipping load")

    print("launching app (Ctrl+C to stop)...")
    run_step(["app.py"])


if __name__ == "__main__":
    main()