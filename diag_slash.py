# python diag_slash.py  — shows every pollster containing a slash, pre-normalisation
import cleaner, pandas as pd
html = cleaner.load_html(cleaner.SOURCES["2019"]["cache"])
tables = cleaner.select_national_tables(html, cleaner.SOURCES["2019"]["years"])
df = cleaner.canonicalise_columns(cleaner.flatten_headers(pd.concat(tables, ignore_index=True)))
raw = df["Pollster"].astype(str)
slashed = sorted(set(raw[raw.str.contains("/", na=False)]))
print(f"{len(slashed)} slashed pollster strings on the 2019 page:\n")
for s in slashed:
    print("   ", repr(s))