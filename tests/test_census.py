import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from aggsyn import datasets
from experiments.census_published_pilot import AGE_BANDS, PUMF_SEX


def test_published_age_bands_cover_each_code_once():
    codes = [code for band in AGE_BANDS.values() for code in band]
    assert codes == list(range(6, 22))
    assert len(codes) == len(set(codes))


def test_census_loader_maps_age_marital_employment_and_gender(monkeypatch):
    rows = []
    for age_code in range(6, 22):
        for gender in [1, 2]:
            rows.append({
                "CMA": 535, "AGEGRP": age_code, "Gender": gender,
                "HDGREE": 1, "MARSTH": 3, "TENUR": 1, "LFACT": 2,
            })
    rows.extend([
        {"CMA": 535, "AGEGRP": 6, "Gender": 1, "HDGREE": 1, "MARSTH": 1, "TENUR": 1, "LFACT": 1},
        {"CMA": 535, "AGEGRP": 6, "Gender": 1, "HDGREE": 1, "MARSTH": 2, "TENUR": 1, "LFACT": 3},
    ])
    source = pd.DataFrame(rows)
    monkeypatch.setattr(datasets.pd, "read_csv", lambda *args, **kwargs: iter([source]))

    loaded = datasets.census("unused")
    expected_age = [0 if code <= 8 else 1 if code <= 10 else 2 if code <= 12
                    else 3 if code <= 14 else 4 if code <= 16 else 5
                    for code in range(6, 22) for _ in [1, 2]]
    assert loaded.age.tolist()[:32] == expected_age
    assert loaded.sex.iloc[0] == 0  # Gender=1 is Woman+
    assert loaded.sex.iloc[1] == 1  # Gender=2 is Man+
    assert loaded.married.iloc[0] == 1  # MARSTH=3 is common-law
    assert loaded.y.iloc[0] == 1  # LFACT=2 is employed
    assert loaded.y.iloc[-1] == 0  # LFACT=3 is not employed


def test_published_columns_align_with_dictionary_gender_codes():
    assert PUMF_SEX == {1: "Gender (3):Women+[3]", 2: "Gender (3):Men+[2]"}
