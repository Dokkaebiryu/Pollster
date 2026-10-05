import textwrap
from pathlib import Path

import pytest

HEADER = """
<tr><th>Date(s) conducted</th><th>Pollster</th><th>Client</th><th>Area</th>
<th>Sample size</th><th>Lab</th><th>Con</th><th>Ref</th><th>LD</th>
<th>Grn</th><th>SNP</th><th>PC</th><th>Others</th></tr>
<tr><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th>
<th></th><th></th><th></th><th></th><th></th></tr>
"""


def _row(date, pollster, client="N/A", area="GB", n="1,500",
         values=("25%", "24%", "23%", "12%", "8%", "3%", "1%", "4%")):
    cells = "".join(f"<td>{v}</td>" for v in values)
    return (f"<tr><td>{date}</td><td>{pollster}</td><td>{client}</td>"
            f"<td>{area}</td><td>{n}</td>{cells}</tr>")


# a row w no numeric party values at all: an embedded event annotation
EVENT_ROW = ("<tr><td>7 Dec</td><td>Partygate scandal begins</td><td></td>"
             "<td></td><td></td><td></td><td></td><td></td><td></td>"
             "<td></td><td></td><td></td><td></td></tr>")

# the GE result: shaped like a poll, carrying a FULL date
RESULT_ROW = _row("4 Jul 2024", "2024 general election", client="–", n="–",
                  values=("33.7%", "23.7%", "14.3%", "12.2%", "6.8%",
                          "2.5%", "0.7%", "6.1%"))


@pytest.fixture
def page_html():
    """a page carrying two wanted year sections and one hypothetical-scenario
    section. this hypothetical table is structurally IDENTICAL to the real
    ones."""
    return textwrap.dedent(f"""
    <html><body>
      <h2>2026</h2>
      <table class="wikitable">
        {HEADER}
        {_row("7–10 Jul", "YouGov")}
        {_row("28 Jun – 1 Jul", "Opinium")}
        {_row("3 Jun", "Savanta ComRes")}
      </table>

      <h2>2025</h2>
      <table class="wikitable">
        {HEADER}
        {_row("30 Dec – 2 Jan", "Techne Archived 12 January 2025 at the Wayback Machine")}
        {_row("14–15 Aug", "YouGov (MRP)", n="47,751")}
        {_row("2–3 Mar", "Deltapoll[permanent dead link]",
              values=("25%", "24%", "-", "12%", "8%", "3%", "-", "4%"))}
        {EVENT_ROW}
      </table>

      <h2>Hypothetical scenarios</h2>
      <table class="wikitable">
        {HEADER}
        {_row("1–2 Feb", "ShouldNotAppear")}
      </table>
    </body></html>
    """)


@pytest.fixture
def page_with_result():
    """a page whose year section opens with the GE result row."""
    return textwrap.dedent(f"""
    <html><body>
      <h2>2024</h2>
      <table class="wikitable">
        {HEADER}
        {RESULT_ROW}
        {_row("2–3 Jul", "More in Common")}
      </table>
    </body></html>
    """)


@pytest.fixture
def alt_header_page():
    """the 2024 election page writes 'Dates conducted', not 'Date(s)
    conducted', which broke the parser."""
    return textwrap.dedent(f"""
    <html><body>
      <h2>2023</h2>
      <table class="wikitable">
        {HEADER.replace("Date(s) conducted", "Dates conducted")}
        {_row("5–6 May", "BMG Research")}
      </table>
    </body></html>
    """)


@pytest.fixture
def cached_page(tmp_path, page_html):
    """writes a fixture page to a temporary cache path."""
    path = tmp_path / "page.html"
    path.write_text(page_html, encoding="utf-8")
    return path