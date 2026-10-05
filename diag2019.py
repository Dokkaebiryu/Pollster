# Run this in your project root: python diag2019.py
import cleaner, pandas as pd
html = cleaner.load_html(cleaner.SOURCES["2019"]["cache"])
tables = cleaner.select_national_tables(html, cleaner.SOURCES["2019"]["years"])
df = cleaner.canonicalise_columns(cleaner.flatten_headers(pd.concat(tables, ignore_index=True)))
cols = list(df.columns)
dupes = [c for c in set(cols) if cols.count(c) > 1]
print("duplicate columns on the 2019 page:", dupes)
print("\nfull column list:")
for c in cols:
    print("  ", repr(c), "  <-- DUP" if cols.count(c) > 1 else "")