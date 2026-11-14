# Data folder

Raw data are NOT included (licensing / privacy). Put the files below in this folder (or pass `--data-dir`).

| Dataset | Files needed | Where from |
|---|---|---|
| `lanark` | `real_lanark.csv` | your administrative records (columns used: age, gender, income_type, has_dependents, outdoor_sleeping, no_income, chronic_homeless) |
| `gss34` | `GSS34PUMFI.txt`, `GSS34PUMFI_i.SAS` | Statistics Canada, General Social Survey cycle 34 (victimisation) PUMF, incident file and its SAS input layout |
| `adult` | `adult-all.csv` | downloaded automatically on first use from a public GitHub copy of the UCI Adult data |
| `mroz`, `gss` | none | shipped inside the `wooldridge` Python package |

Run any experiment on a subset with `--datasets lanark mroz`.
