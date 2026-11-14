# Data folder

Raw data are NOT included (licensing / privacy). Put the files below in this folder (or pass `--data-dir`).

| Dataset | Files needed | Where from |
|---|---|---|
| `lanark` | `real_lanark.csv` | your administrative records (columns used: age, gender, income_type, has_dependents, outdoor_sleeping, no_income, chronic_homeless) |
| `gss34` | `GSS34PUMFI.txt`, `GSS34PUMFI_i.SAS` | Statistics Canada, General Social Survey cycle 34 (victimisation) PUMF, incident file and its SAS input layout |
| `adult` | `adult-all.csv` | downloaded automatically on first use from a public GitHub copy of the UCI Adult data |
| `mroz`, `gss` | none | shipped inside the `wooldridge` Python package |
| `census` | `data_donnees_2021_ind.csv` | 2021 Census individual PUMF; loader keeps Toronto CMA (`CMA=535`) persons aged 15+ and ignores `WEIGHT` |

Run any experiment on a subset with `--datasets lanark mroz`.

The Census loader uses `LFACT` 1 or 2 as employed, drops nonresponse codes,
maps `Gender=1` to Woman+ and `Gender=2` to Man+, and maps `AGEGRP` 6--21
(ages 15+) to the published-compatible 15--24, 25--34, 35--44, 45--54,
55--64 and 65+ bands. It also collapses education, marital status and tenure
into small integer-coded categories. The accompanying Toronto labour-status table
(`toronto-cma-98-10-0486-01-lfs-age-gender.csv`) is a published aggregate
reference, not currently used as an input: its 75 count cells per sex column
are multiples of 5, while its rate rows are percentages. It does not contain
all six features required by the experiments.

The published-table pilot does use this aggregate file, but only for age,
gender and employment-status counts; it does not use PUMF rows to construct
fitting tables. Its synthetic-data model uses the PUMF only for a weighted
held-out test and a separate weighted training baseline.
