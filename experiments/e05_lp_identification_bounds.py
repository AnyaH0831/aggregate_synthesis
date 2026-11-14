"""E05 - what the published tables cannot determine: exact worst-case bounds on subgroup outcome rates.

For subgroups defined by two feature values (a combination that is NOT published together with the outcome), a linear-fractional
program (Charnes-Cooper) gives the lowest and highest outcome rate over ALL joint tables consistent with the published margins.
Scenarios:  S2 (totals + outcome x feature),  S3 (totals + all pairs),  S3r (S3 rounded to 10 with cells < 10 hidden, N = 500).
Reported: width of the possible range, the max-entropy estimate's error, and whether the subgroup can be CERTIFIED above/below the base rate.
Output: results/e05_lp_identification_bounds.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from aggsyn import TableSpace, Selector, datasets, subgroup_rate_bounds
from aggsyn.cli import make_parser, setup

p = make_parser(__doc__)
p.add_argument("--subgroups", type=int, default=24)
args = setup(p.parse_args())
nsub = 3 if args.quick else args.subgroups
N = 500
rows = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir); s = TableSpace(d); S = Selector(s, tol=1e-10, maxit=1500)
    t = s.table(d); p_ = t / t.sum(); L = s.L; base = p_[..., 1].sum()
    rng = np.random.default_rng(0)
    cand = []
    import itertools
    for i, j in itertools.combinations(range(L), 2):
        for a in range(s.dims[i]):
            for b in range(s.dims[j]):
                m = (s.idx[i] == a) & (s.idx[j] == b)
                sz = p_.ravel()[m].sum()
                if sz >= 0.02 and p_.ravel()[m & (s.idx[L] == 1)].sum() > 0:
                    cand.append((i, j, a, b, sz))
    pick = [cand[k] for k in rng.choice(len(cand), min(nsub, len(cand)), replace=False)]
    for sc, tabs in [("S2", s.tabs("S2")), ("S3", s.tabs("S3")), ("S3r", s.tabs("S3"))]:
        m_exact = s.flat_margins(p_, tabs)
        q = S.fit(p_, tabs).ravel()
        if sc == "S3r":
            c = N * m_exact; sup = c < 10; r = 10 * np.round(c / 10)
            lo = np.where(sup, 0, np.maximum(r - 5, 0)) / N; hi = np.where(sup, 9, r + 5) / N
        else:
            lo = hi = m_exact
        for (i, j, a, b, sz) in pick:
            mask = (s.idx[i] == a) & (s.idx[j] == b)
            rate_lo, rate_hi = subgroup_rate_bounds(s, tabs, lo, hi, mask)
            true = p_.ravel()[mask & (s.idx[L] == 1)].sum() / sz
            est = q[mask & (s.idx[L] == 1)].sum() / q[mask].sum()
            rows.append(dict(data=name, scen=sc, sub=f"{i},{j},{a},{b}", size=sz, true=true, lo=rate_lo, hi=rate_hi, width=rate_hi - rate_lo,
                             maxent=est, base=base, inside=bool(rate_lo - 1e-7 <= true <= rate_hi + 1e-7)))
        print(name, sc, flush=True)
r = pd.DataFrame(rows)
r.to_csv(Path(args.out_dir) / "e05_lp_identification_bounds.csv", index=False)
r["err"] = (r.maxent - r.true).abs(); r["certified"] = (r.lo > r.base) | (r.hi < r.base)
print("truth inside bounds in all cases:", bool(r.inside.all()))
print(r.groupby("scen").agg(width=("width", "mean"), err=("err", "mean"), certified=("certified", "mean")).round(3))
