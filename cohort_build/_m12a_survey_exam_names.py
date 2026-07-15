# -*- coding: utf-8 -*-
"""M1.2a: stream-read MIMIC-IV-Note radiology_detail to survey primary exam_name."""
import gzip, io, csv, time, json
from collections import Counter
import paths

primary_names = Counter()
all_names = Counter()
primary_codes = Counter()
all_field_names = Counter()
total_rows = 0
t0 = time.time()

print(f"[ENV={paths.ENV}] Streaming radiology_detail gzip...", flush=True)

with paths.open_radiology_detail_gz() as f:
    with gzip.GzipFile(fileobj=f) as gz:
        reader = csv.DictReader(io.TextIOWrapper(gz, encoding="utf-8"))
        for row in reader:
            total_rows += 1
            fn = row.get("field_name", "")
            fo = row.get("field_ordinal", "")
            fv = row.get("field_value", "")
            all_field_names[fn] += 1
            if fn == "exam_name":
                all_names[fv] += 1
                if fo == "1":
                    primary_names[fv] += 1
            elif fn == "exam_code":
                if fo == "1":
                    primary_codes[fv] += 1
            if total_rows % 1000000 == 0:
                print(f"  ... {total_rows:,} rows, elapsed={time.time()-t0:.1f}s", flush=True)

elapsed = time.time() - t0
print(f"\n=== Summary ===")
print(f"Total radiology_detail rows: {total_rows:,}")
print(f"Elapsed: {elapsed:.1f}s")
print(f"\nfield_name distribution:")
for k, v in all_field_names.most_common():
    print(f"  {v:>10,}  {k}")
print(f"\nDistinct exam_name (any ordinal): {len(all_names):,}")
print(f"Distinct PRIMARY exam_name (ordinal=1): {len(primary_names):,}")
print(f"Total primary exam_name occurrences: {sum(primary_names.values()):,}")
print(f"Distinct PRIMARY exam_code (ordinal=1): {len(primary_codes):,}")
print(f"\n=== Top 80 PRIMARY exam_name ===")
for name, cnt in primary_names.most_common(80):
    print(f"  {cnt:>7,}  {name}")
print(f"\n=== Top 40 PRIMARY exam_code ===")
for code, cnt in primary_codes.most_common(40):
    print(f"  {cnt:>7,}  {code}")

with open(paths.PRIMARY_EXAM_NAMES_DIST_JSON, "w", encoding="utf-8") as f:
    json.dump(primary_names.most_common(), f, ensure_ascii=False, indent=2)
with open(paths.PRIMARY_EXAM_CODES_DIST_JSON, "w", encoding="utf-8") as f:
    json.dump(primary_codes.most_common(), f, ensure_ascii=False, indent=2)
print(f"\nSaved: {paths.PRIMARY_EXAM_NAMES_DIST_JSON}")
print(f"Saved: {paths.PRIMARY_EXAM_CODES_DIST_JSON}")
