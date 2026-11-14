"""Pilot using only the published Toronto age-by-gender labour table.

The published count estimates are checked under conservative +/-5 intervals.
This is not asserted to be an exact interval for weighted estimates, and the
intervals contain no sampling uncertainty. The script first checks that status
cells and independently rounded total rows can be satisfied together, then
chooses one feasible integer table and trains on synthetic rows from it.
PUMF rows are used only for the held-out evaluation and separate baseline.
"""
import argparse
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
    "15-24": [6, 7, 8],
    "25-34": [9, 10],
    "35-44": [11, 12],
    "45-54": [13, 14],
    "55-64": [15, 16],
    "65-74": [17, 18],
    "75+": [19, 20, 21],
}
PUMF_SEX = {1: "Gender (3):Women+[3]", 2: "Gender (3):Men+[2]"}


def published_counts(path=PUBLISHED):
    d = pd.read_csv(path)
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


def feasible_status_table(pub, published_path=PUBLISHED):
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
    d = pd.read_csv(published_path)
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
    if result.x is None or not result.success:
        return None, n, len(rows), result.status, result.message
    return {key: int(round(result.x[i])) for key, i in index.items()}, n, len(rows), result.status, result.message


def load_pumf(path=PUMF):
    columns = ["CMA", "AGEGRP", "Gender", "LFACT", "WEIGHT"]
    parts = []
    for d in pd.read_csv(path, usecols=columns, chunksize=250_000):
        d = d[(d["CMA"] == 535) & d["AGEGRP"].between(6, 21)]
        d = d[d["Gender"].isin([1, 2]) & d["LFACT"].between(1, 14)]
        if not d.empty:
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    band_for_code = {code: band for band, codes in AGE_BANDS.items() for code in codes}
    age_index = {band: i for i, band in enumerate(AGE_BANDS)}
    return pd.DataFrame({
        "age": d["AGEGRP"].map(band_for_code).map(age_index).astype(int),
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


def diagnostics(synthetic, model_pub, model_pumf, test, table):
    age_index = {band: i for i, band in enumerate(AGE_BANDS)}
    rows = []
    for band, age in age_index.items():
        for sex in [1, 2]:
            cell = test[(test.age == age) & (test.sex == int(sex == 2))]
            fitted = {
                status: table[(band, sex, status)]
                for status in STATUS
            }
            total = fitted["Employed"] + fitted["Unemployed"] + fitted["Not in the labour force"]
            pub_rate = fitted["Employed"] / total if total else np.nan
            weight = cell.weight.sum()
            weighted_rate = np.average(cell.y, weights=cell.weight) if weight else np.nan
            pred_pub = model_pub.predict_proba(pd.DataFrame({"age": [age], "sex": [int(sex == 2)]}))[:, 1][0]
            pred_pumf = model_pumf.predict_proba(pd.DataFrame({"age": [age], "sex": [int(sex == 2)]}))[:, 1][0]
            signed = pub_rate - weighted_rate if np.isfinite(weighted_rate) else np.nan
            rows.append({
                "age_band": band,
                "gender": "Woman+" if sex == 1 else "Man+",
                "fitted_employment_count": fitted["Employed"],
                "fitted_nonemployment_count": fitted["Unemployed"] + fitted["Not in the labour force"],
                "fitted_employment_rate": pub_rate,
                "published_model_rate": pred_pub,
                "pumf_model_rate": pred_pumf,
                "heldout_pumf_record_count": len(cell),
                "heldout_pumf_weight_sum": weight,
                "heldout_weighted_employment_rate": weighted_rate,
                "signed_rate_difference": signed,
                "absolute_rate_difference": abs(signed) if np.isfinite(signed) else np.nan,
                "empty_or_zero_weight_test_cell": len(cell) == 0 or weight <= 0,
            })
    return pd.DataFrame(rows)


def joint_tvd(synthetic, test):
    cols = ["age", "sex", "y"]
    a = synthetic.groupby(cols).size().reindex(
        pd.MultiIndex.from_product([range(len(AGE_BANDS)), [0, 1], [0, 1]]), fill_value=0
    ).to_numpy(dtype=float)
    b = test.assign(weighted_y=test.weight * test.y, weighted_nontest=test.weight * (1 - test.y))
    b0 = b.groupby(["age", "sex"])[["weighted_nontest", "weighted_y"]].sum()
    b = b0.reindex(pd.MultiIndex.from_product([range(len(AGE_BANDS)), [0, 1]]), fill_value=0).to_numpy().ravel()
    return 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args(argv)
    published_path = Path(args.data_dir) / PUBLISHED.name
    pumf_path = Path(args.data_dir) / PUMF.name
    pub = published_counts(published_path)
    table, variables, constraints, solver_status, solver_message = feasible_status_table(pub, published_path)
    print(f"published_constraints_feasible={table is not None}")
    print(f"solver_status={solver_status} solver_message={solver_message}")
    print(f"integer_variables={variables} checked_constraints={constraints}")
    if table is None:
        raise SystemExit("Published random-rounding constraints are infeasible")

    synthetic = synthetic_rows(table)
    pumf = load_pumf(pumf_path)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(split.split(pumf, pumf.y))
    train, test = pumf.iloc[train_idx], pumf.iloc[test_idx]

    model_pub = LogisticRegression(max_iter=3000).fit(synthetic[["age", "sex"]], synthetic.y)
    model_pumf = LogisticRegression(max_iter=3000).fit(
        train[["age", "sex"]], train.y, sample_weight=train.weight
    )
    cell = diagnostics(synthetic, model_pub, model_pumf, test, table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out_dir / "census_published_pilot_cells.csv", index=False)
    max_diff = cell.absolute_rate_difference.max()
    result = pd.DataFrame([{
        "seed": args.seed,
        "age_bands": "|".join(AGE_BANDS),
        "weighting": "WEIGHT for PUMF baseline and held-out evaluation only",
        "rounding_assumption": "conservative reported +/- 5; no sampling uncertainty",
        "solver_status": solver_status,
        "solver_message": solver_message,
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
        "max_absolute_cell_rate_difference": max_diff,
        "joint_tvd_28_cells": joint_tvd(synthetic, test),
        "empty_or_zero_weight_test_cells": int(cell.empty_or_zero_weight_test_cell.sum()),
    }])
    result.to_csv(out_dir / "census_published_pilot.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
