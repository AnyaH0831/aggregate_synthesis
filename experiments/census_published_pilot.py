"""Pilot using only the published Toronto age-by-gender labour table.

The published counts are treated as StatCan random-rounded values: each
reported multiple of five permits a true count in [reported - 5, reported + 5].
The script first checks that the status cells and their published total rows can
be satisfied together, then chooses one feasible integer table and trains on
synthetic rows from it. PUMF rows are used only for the held-out evaluation and
the separate PUMF-training baseline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


DATA_DIR = Path("data")
PUBLISHED = DATA_DIR / "toronto-cma-98-10-0486-01-lfs-age-gender.csv"
PUMF = DATA_DIR / "data_donnees_2021_ind.csv"
STATUS = ["Employed", "Unemployed", "Not in the labour force"]
AGE_BANDS = {
    "15-24": [6, 7],
    "25-34": [8, 9],
    "35-44": [10, 11],
    "45-54": [12, 13],
    "55-64": [14, 15],
    "65-74": [16, 17],
    "75+": [18, 19, 20, 21],
}
PUMF_SEX = {1: "Gender (3):Women+[3]", 2: "Gender (3):Men+[2]"}


def published_counts():
    d = pd.read_csv(PUBLISHED)
    d["Labour force status (8)"] = d["Labour force status (8)"].str.strip()
    d["Age (15A)"] = d["Age (15A)"].str.strip()
    labels = {
        "15-24": "15 to 24 years",
        "25-34": "25 to 34 years",
        "35-44": "35 to 44 years",
        "45-54": "45 to 54 years",
        "55-64": "55 to 64 years",
        "65-74": "65 to 74 years",
        "75+": "75 years and over",
    }
    out = {}
    for band, label in labels.items():
        for sex, column in PUMF_SEX.items():
            for status in STATUS:
                row = d[(d["Labour force status (8)"] == status) & (d["Age (15A)"] == label)]
                if len(row) != 1:
                    raise ValueError(f"Expected one published row for {status}, {label}")
                out[(band, sex, status)] = int(row.iloc[0][column])
            row = d[
                (d["Labour force status (8)"] == "Total - Labour force status")
                & (d["Age (15A)"] == label)
            ]
            out[(band, sex, "Total")] = int(row.iloc[0][column])
    return out


def random_rounding_bounds(values, radius=5):
    values = np.asarray(values, dtype=float)
    return np.maximum(0, values - radius), values + radius


def feasible_status_table(pub):
    keys = [(band, sex, status) for band in AGE_BANDS for sex in [1, 2] for status in STATUS]
    index = {key: i for i, key in enumerate(keys)}
    n = len(keys)
    rows, lower, upper = [], [], []

    def add(row, lo, hi):
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    for key in keys:
        row = np.zeros(n)
        row[index[key]] = 1
        lo, hi = random_rounding_bounds([pub[key]])
        add(row, lo[0], hi[0])

    for band in AGE_BANDS:
        for sex in [1, 2]:
            row = np.zeros(n)
            for status in STATUS:
                row[index[(band, sex, status)]] = 1
            lo, hi = random_rounding_bounds([pub[(band, sex, "Total")]])
            add(row, lo[0], hi[0])

    # Include the published Total - Age row as an additional consistency check.
    d = pd.read_csv(PUBLISHED)
    d["Labour force status (8)"] = d["Labour force status (8)"].str.strip()
    d["Age (15A)"] = d["Age (15A)"].str.strip()
    for sex, column in PUMF_SEX.items():
        for status in STATUS:
            row = np.zeros(n)
            for band in AGE_BANDS:
                row[index[(band, sex, status)]] = 1
            value = d[
                (d["Labour force status (8)"] == status)
                & (d["Age (15A)"] == "Total - Age")
            ].iloc[0][column]
            lo, hi = random_rounding_bounds([value])
            add(row, lo[0], hi[0])

    A = np.vstack(rows)
    result = milp(
        np.zeros(n),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.full(n, np.inf)),
        constraints=LinearConstraint(A, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30},
    )
    if result.x is None:
        return None, n, len(rows)
    return {key: int(round(result.x[i])) for key, i in index.items()}, n, len(rows)


def load_pumf():
    columns = ["CMA", "AGEGRP", "Gender", "LFACT", "WEIGHT"]
    parts = []
    for d in pd.read_csv(PUMF, usecols=columns, chunksize=250_000):
        d = d[(d["CMA"] == 535) & d["AGEGRP"].between(6, 21)]
        d = d[d["Gender"].isin([1, 2]) & d["LFACT"].between(1, 14)]
        if not d.empty:
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    band_for_code = {code: band for band, codes in AGE_BANDS.items() for code in codes}
    return pd.DataFrame({
        "age": d["AGEGRP"].map(band_for_code).astype("category").cat.codes,
        "sex": (d["Gender"] == 2).astype(int),
        "y": d["LFACT"].isin([1, 2]).astype(int),
        "weight": d["WEIGHT"].astype(float),
    })


def synthetic_rows(table):
    rows = []
    age_index = {band: i for i, band in enumerate(AGE_BANDS)}
    for (band, sex, status), count in table.items():
        rows.extend([(age_index[band], int(sex == 2), int(status == "Employed"))] * count)
    return pd.DataFrame(rows, columns=["age", "sex", "y"])


def main():
    pub = published_counts()
    table, variables, constraints = feasible_status_table(pub)
    print(f"published_constraints_feasible={table is not None}")
    print(f"integer_variables={variables} checked_constraints={constraints}")
    if table is None:
        raise SystemExit("Published random-rounding constraints are infeasible")

    synthetic = synthetic_rows(table)
    pumf = load_pumf()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2021)
    train_idx, test_idx = next(split.split(pumf, pumf.y))
    train, test = pumf.iloc[train_idx], pumf.iloc[test_idx]

    model_pub = LogisticRegression(max_iter=3000).fit(synthetic[["age", "sex"]], synthetic.y)
    model_pumf = LogisticRegression(max_iter=3000).fit(
        train[["age", "sex"]], train.y, sample_weight=train.weight
    )
    result = pd.DataFrame([{
        "synthetic_rows": len(synthetic),
        "pumf_rows": len(pumf),
        "weighted_test_population": test.weight.sum(),
        "auc_published_synthetic": roc_auc_score(
            test.y, model_pub.predict_proba(test[["age", "sex"]])[:, 1],
            sample_weight=test.weight,
        ),
        "auc_pumf_train": roc_auc_score(
            test.y, model_pumf.predict_proba(test[["age", "sex"]])[:, 1],
            sample_weight=test.weight,
        ),
    }])
    result.to_csv("results/census_published_pilot.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
