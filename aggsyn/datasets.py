"""Dataset loaders. Every loader returns an integer-coded DataFrame whose LAST column is the binary outcome ``y``.

Datasets
  lanark  real administrative homelessness records (you supply data/real_lanark.csv); y = chronic_homeless
  adult   1994 US Census extract (downloaded once from GitHub into data/adult-all.csv); y = income > 50K
  mroz    Mroz (1987) women's labour survey, from the ``wooldridge`` package; y = in labour force
  gss     US General Social Survey 1994-2006 (``wooldridge`` 'happiness'); y = "very happy"
  gss34   Statistics Canada GSS cycle 34 victimisation INCIDENT file (you supply the txt + SAS layout); y = police found out
  census  2021 Canadian Census PUMF, Toronto CMA; y = employed
"""
import os
import re
import urllib.request
import numpy as np
import pandas as pd

DATASETS = ["lanark", "adult", "mroz", "gss", "gss34", "census"]
# training-set size used in the sample-level experiments (Lanark: the paper's 80% split of 567 rows)
TRAIN_SIZES = {"lanark": 454, "adult": 500, "mroz": 600, "gss": 500, "gss34": 500, "census": 500}
ADULT_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"


def lanark(data_dir="data"):
    d = pd.read_csv(os.path.join(data_dir, "real_lanark.csv"))
    d = d[d.gender.isin(["male", "female"])].copy()          # drops 1 row with a single non-binary entry
    d["age"] = d.age.fillna(d.age.median())
    return pd.DataFrame({
        "age": pd.cut(d.age, [0, 24, 34, 44, 54, 64, 200], labels=False),
        "gender": (d.gender == "male").astype(int),
        "income": d.income_type.fillna("NA").map({"NA": 0, "welfare": 1, "disability": 2, "employment": 3, "other": 4, "informal": 5}),
        "dep": d.has_dependents, "outdoor": d.outdoor_sleeping, "noinc": d.no_income,
        "y": d.chronic_homeless}).astype(int).reset_index(drop=True)


def adult(data_dir="data"):
    path = os.path.join(data_dir, "adult-all.csv")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(ADULT_URL, path)
    a = pd.read_csv(path, header=None)
    a.columns = ["age", "wc", "fn", "edu", "eduN", "marital", "occ", "rel", "race", "sex", "cg", "cl", "hours", "country", "inc"]
    return pd.DataFrame({
        "age": pd.cut(a.age, [0, 24, 34, 44, 54, 64, 100], labels=False),
        "sex": (a.sex.str.strip() == "Male").astype(int),
        "edu": pd.cut(a.eduN, [0, 8, 9, 12, 16], labels=False),
        "marital": a.marital.str.strip().map(lambda s: 0 if s.startswith("Married-civ") else (1 if s == "Never-married" else 2)),
        "hours": pd.cut(a.hours, [0, 34, 40, 100], labels=False),
        "race": (a.race.str.strip() == "White").astype(int),
        "y": (a.inc.str.strip() == ">50K").astype(int)}).astype(int)


def mroz(data_dir="data"):
    import wooldridge as w
    m = w.data("mroz")
    return pd.DataFrame({
        "age": pd.cut(m.age, [0, 35, 42, 49, 100], labels=False),
        "educ": pd.cut(m.educ, [0, 11, 12, 15, 30], labels=False),
        "kidslt6": m.kidslt6.clip(0, 2), "kidsge6": m.kidsge6.clip(0, 2),
        "huseduc": pd.cut(m.huseduc, [-1, 11, 12, 15, 30], labels=False),
        "y": m.inlf}).astype(int)


def gss(data_dir="data"):
    import wooldridge as w
    h = w.data("happiness").dropna(subset=["vhappy", "educ", "income", "attend", "female", "black", "divorce"]).copy()
    yes = lambda s: s.astype(str).str.lower().isin(["yes", "1", "1.0"]).astype(int)
    inc = h.income.astype(str).str.extract(r"(\d+)")[0].astype(float).fillna(0)

    def att(v):
        v = str(v).lower()
        return 0 if ("never" in v or v.startswith("lt")) else 1 if ("once a year" in v or "sevrl" in v) else 2 if ("month" in v or "mnth" in v) else 3

    return pd.DataFrame({
        "educ": pd.cut(h.educ, [-1, 11, 12, 15, 30], labels=False),
        "income": pd.cut(inc, [-1, 9999, 19999, 24999, 1e9], labels=False),   # top-coded at $25,000+
        "attend": h.attend.map(att), "female": yes(h.female), "black": yes(h.black), "divorce": yes(h.divorce),
        "y": yes(h.vhappy)}).astype(int)


def read_sas_fixed_width(txt_path, sas_input_path):
    """Read a Statistics Canada fixed-width file using the '@ start NAME width.dec' lines of the SAS input file."""
    spec = []
    for line in open(sas_input_path, encoding="latin-1"):
        m = re.match(r"\s*@\s+(\d+)\s+(\S+)\s+(\d+)\.(\d*)\s", line)
        if m:
            start = int(m.group(1)) - 1
            spec.append((m.group(2), start, start + int(m.group(3))))
    return pd.read_fwf(txt_path, colspecs=[(a, b) for _, a, b in spec], names=[n for n, _, _ in spec], header=None, dtype=str)


def gss34(data_dir="data", txt="GSS34PUMFI.txt", layout="GSS34PUMFI_i.SAS"):
    """Outcome: PFO_100 'police found out about the incident' (1 = yes, 2 = no; 9 not stated is dropped).
    Features (collapsed): incident type, location, weapon present, victim present, offender relationship, cost of loss."""
    d = read_sas_fixed_width(os.path.join(data_dir, txt), os.path.join(data_dir, layout))
    for c in d.columns[3:66]:
        d[c] = d[c].str.strip()
    d = d[d.PFO_100.isin(["1", "2"]) & (d.WHR_100 != "9")].copy()
    ms = d.MSCRIMEC.astype(int)
    typ = np.select([ms.isin([101, 202, 203, 304]), ms.isin([405, 406]), ms.isin([507, 508, 509])], [0, 1, 2], 3)
    return pd.DataFrame({
        "type": typ,
        "loc": d.WHR_100.map({"1": 0, "2": 1, "5": 1, "3": 2, "4": 3}).values,
        "weapon": d.CIR_100.map({"1": 0, "2": 1}).fillna(2).values,
        "present": d.CIR_090.map({"1": 0, "2": 1}).fillna(2).values,
        "offender": d.OFFENDRC.map({"1": 0, "2": 0, "3": 1}).fillna(2).values,
        "cost": d.TOTCOST.map({"01": 0, "02": 1, "03": 1, "04": 1, "05": 2, "06": 2}).fillna(3).values,
        "y": (d.PFO_100 == "1").astype(int).values}).astype(int).reset_index(drop=True)


def census(data_dir="data", filename="data_donnees_2021_ind.csv"):
    """Load an unweighted, integer-coded Toronto 2021 Census PUMF sample.

    The PUMF's ``WEIGHT`` column is deliberately ignored: this first case study
    treats each sampled row as one person.  We keep persons aged 15+ in CMA 535,
    drop Census nonresponse codes, and use ``LFACT`` 1 or 2 (employed) as ``y``.
    ``AGEGRP`` 6--21 are the 15+ codes; ``Gender=1`` is Woman+ and ``Gender=2``
    is Man+.
    """
    path = os.path.join(data_dir, filename)
    columns = ["CMA", "AGEGRP", "Gender", "HDGREE", "MARSTH", "TENUR", "LFACT"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        chunk = chunk[(chunk["CMA"] == 535) & chunk["AGEGRP"].between(6, 21)]
        chunk = chunk[
            chunk["Gender"].isin([1, 2])
            & chunk["HDGREE"].between(1, 14)
            & chunk["MARSTH"].isin([1, 2, 3, 4, 5, 6])
            & chunk["TENUR"].isin([1, 2])
            & chunk["LFACT"].between(1, 14)
        ]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No valid Toronto Census rows found in {path}")
    d = pd.concat(chunks, ignore_index=True)
    age = pd.cut(d["AGEGRP"], [5, 7, 9, 11, 13, 15, 21], labels=False)
    education = np.select(
        [d["HDGREE"].isin([1, 2]), d["HDGREE"].isin([3, 4, 5, 6])],
        [0, 1],
        default=2,
    )
    return pd.DataFrame({
        "age": age,
        "sex": (d["Gender"] == 2).astype(int),
        "education": education,
        "married": d["MARSTH"].isin([1, 2]).astype(int),
        "tenure": (d["TENUR"] == 1).astype(int),
        "y": d["LFACT"].isin([1, 2]).astype(int),
    }).astype(int).reset_index(drop=True)


_LOADERS = {"lanark": lanark, "adult": adult, "mroz": mroz, "gss": gss, "gss34": gss34, "census": census}


def load(name, data_dir="data"):
    return _LOADERS[name](data_dir)
