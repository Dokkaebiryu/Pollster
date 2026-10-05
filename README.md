# Pollster

A UK Westminster voting-intention poll aggregator. It collects published opinion polls,
models the underlying trend per-party with sample-size-weighted local regression, and
draws an uncertainty band around each estimate - so that the picture it presents is a
PoP showing both trend *and* confidence levels according to the data.

Built as a BSc Computer Science final-year project, achieving **90%**.

May be converted to React when I have the time, to optimise load times and interactivity. 

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Dash](https://img.shields.io/badge/dash-plotly-brightgreen)
![Tests](https://img.shields.io/badge/tests-81%20passing-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What it does

- **Aggregates** UK voting-intention polls (2016-2026) scraped from Wikipedia's
  general-election polling pages.
- **Models** each party's trend with a weighted LOESS:
  polls are weighted by sample size, the bandwidth adapts to polling density,
  and the fit is constrained so it cannot extrapolate beyond the data observed.
- **Quantifies uncertainty**: the shaded band is derived from the
  locally-typical sample size, and represents sampling error as a *lower bound*
  on true uncertainty, not a false-precision figure. Important not to overstate confidence.
- **Distinguishes absence from zero** throughout: a party a pollster did not
  prompt for is recorded as missing, never as 0%, so it is excluded from the
  model rather than dragging the trend down.
- **Surfaces provenance** - every poll carries its source, fieldwork dates,
  sample size and methodology, visible in the interface rather than buried.

The interface is an interactive dashboard: the timeline can be zoomed in on, parties isolated,
pollsters filtered by, dates pinned to inspect polls in fieldwork, net change read between two dates,
and political events seen marked on the timeline.

---

## Running it

Requires **Python 3.10+** and a browser window at least **900px wide** for legibility.
Opening it on a window below this size will redirect. 

```bash
pip install -r requirements.txt
python run.py
```

Then open the URL shown in the terminal (default `http://127.0.0.1:8050`).

`run.py` is a single-command launcher: it checks each Wikipedia source for edits,
re-fetches *only* what has changed, rebuilds the database if needed, and starts the
app. On a first run it fetches everything; on later runs it serves from cache
unless a source has actually been updated.

```bash
python run.py              # refresh what changed, then serve
python run.py --offline    # skip all network checks, serve cached data
python run.py --refresh    # force a full re-fetch and rebuild
```

If the database already exists and you just want to serve it, you can run the
app directly:

```bash
python app.py
```

---

## Testing

```bash
pytest
```

81 tests cover the cleaning, storage, and analysis layers, and the pure control
logic extracted from the interface callbacks.

---

## How it's built

A five-layer pipeline, each layer independently testable and touching only the
one next to it:

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Acquisition | `fetch_page.py` | Fetch and cache the source pages |
| Cleaning | `cleaner.py` | Parse HTML tables; normalise pollsters, parties, dates; separate events from polls |
| Storage | `database.py`, `loader.py` | Relational schema (SQLAlchemy) and the wide→long load into SQLite |
| Analysis | `analyser.py`, `trend.py` | Read back into analysable shape; weighted-LOESS trends and confidence bands |
| Presentation | `app.py`, `assets/` | Plotly Dash dashboard |

`run.py` orchestrates the first four; `app.py` is the running application.

**Stack:** Python, pandas, NumPy, Plotly Dash, SQLAlchemy, SQLite,
BeautifulSoup, pytest.

---

## Data and licensing

**Code** is released under the [MIT License](LICENSE).

**Polling data** is not distributed with this repository. It is scraped at
runtime from Wikipedia, which licenses its content under
[CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/); the database and
cached pages are rebuilt locally by `run.py` and are intentionally excluded from
version control. Polling figures originate with the named polling organisations
and the outlets that commissioned them.

This tool is an independent educational project and is not affiliated with any
polling organisation, political party or news outlet.
