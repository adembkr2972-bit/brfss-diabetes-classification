# 2024 CDC source verification

## Conclusion

**Conclusion: identical XPT; CSV numerically equivalent for all 68 used fields within serialization tolerance.** The SHA-256 of the freshly downloaded CDC 2024 XPT is 7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c; the local companion XPT SHA-256 is 7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c. The byte counts are 1,093,874,240 and 1,093,874,240. These XPT files are byte-for-byte identical. Exact IEEE comparisons of the CSV and XPT representation found 203,671 rows with at least one difference; all such differences were extremely small floating-point values that the CSV serializes as zero. The affected values were `5.397605346934028e-79` in the XPT reader versus `0` in the CSV, in `_DRNKWK3` and one `CHILDREN` observation. With absolute tolerance `1e-75`, 0 rows differed across all 68 used fields. This is consistent with serialization precision, while the two XPTs themselves match byte-for-byte.

## File and schema counts

| File | Rows | Columns | Columns only in this file | SHA-256 |
|---|---:|---:|---|---|
| CDC 2024 official XPT | 457,670 | 301 | None | `7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c` |
| Local companion XPT | Same as official because SHA-256 matches | 301 | None | `7d53cf088f858e02bcc029a51cf0408cc68f40a0ebeb02a3c0f73058aab36b1c` |
| Local analysis CSV | 457,670 | 301 | None | `15b95a4c8b751eb3a6da9e97f4fd69f1b48b77b95663d1b0c95822c69d31e4fa` |

The analysis used 67 predictors plus `DIABETE4` (68 fields). All used fields were present in the official XPT. Per-field exact and tolerance-based mismatch counts are in `csv_to_official_xpt_field_comparison.csv`.

The saved CDC layout audit (`outputs/reports/source_layout_audit.md`, layout HTML SHA-256 `8c823d19f09bc02aa16b40c86f4beb29ca388e5d6cf26c197f7074adc8a43530`) found 296 names shared with the 301-column experiment CSV. Five CSV names were absent from that 300-row layout: `RCSGEND1`, `RCSXBRTH`, `_CLLCPWT`, `_LCSPSTF`, and `_LLCPWT`. Four layout names were absent from the CSV: `LCSFIRS_`, `LCSLAST`, `_CLCWTV2`, and `_LCPWTV2`. All 68 analysis fields are in the shared-name set; none of these nine names is used. This explains the one-column net difference between those two saved headers, but not why the annual page states 345 variables while its linked XPT has 301 columns.

## CDC layout count discrepancy

The CDC annual 2024 page states that the SAS transport file contains 345 variables. The freshly downloaded XPT archive contains a 1,093,874,240-byte file with 301 columns. The local companion XPT has the same SHA-256 and column count. The existing project audit parsed 300 names from the CDC variable layout, and the downloaded 2024 codebook contains 297 SAS variable-name sections. These materials do not document the exact reason that the page's 345-variable statement differs from the linked XPT payload's 301 columns. The discrepancy remains unresolved; it does not affect the 68-field comparison above. The CDC page's modification note does not establish the cause of this count difference.

## Download provenance

The official XPT ZIP was downloaded from the data-file link on the [CDC 2024 BRFSS page](https://www.cdc.gov/brfss/annual_data/annual_2024.html). Download ZIP: `LLCP2024XPT.zip`, 83,007,956 bytes, SHA-256 `d33de91a584e866f657d29b1d5b51b5672467286ed22d021e36ee3c7b4d712a4`. The extracted payload SHA-256 and exact local match are recorded above. The full command, runtime versions, paths, and input checksums are in `run_metadata.json`.
