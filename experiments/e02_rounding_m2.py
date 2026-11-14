"""E02 - realistic public tables (rounded to b, cells below b suppressed): standard fitting vs interval fitting (M2).

 IPF-naive : treat rounded numbers as exact (suppressed cells = 2), IPF, controlled rounding to the midpoint targets.
 M2        : maximum entropy subject to the RANGES [r - b/2, r + b/2] (suppressed: [0, b-1]).
Tested at several rounding bases b on all-pairs tables (S3). Result in the paper: M2 helps only when rounding is coarse (b=20).
Output: results/e02_rounding_m2.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from aggsyn import TableSpace, datasets
from aggsyn.cli import make_parser, setup

p = make_parser(__doc__, default_datasets=["lanark"], default_reps=6)
p.add_argument("--bases", nargs="+", type=int, default=[5, 10, 20])
args = setup(p.parse_args())
rows = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir)
    s = TableSpace(d)
    for b in args.bases:
        for rep, (tr, te) in enumerate(StratifiedShuffleSplit(args.reps, test_size=0.2, random_state=0).split(d, d.y)):
            dtr = d.iloc[tr].reset_index(drop=True); N = len(dtr); truth = s.table(dtr)
            tabs = s.tabs("S3")
            pub = s.publish(truth, tabs, b); lo, hi = s.intervals(pub, b)
            pn = s.ipf(tabs, s.naive_targets(pub), N, 300)
            pm = s.m2(tabs, lo, hi, N)
            for m, pp in [("IPF-naive+ILP", pn), ("M2+ILP", pm)]:
                syn = s.ilp(pp, N, tabs, lo, hi, interval=False, tl=3 if args.quick else 6)
                rows.append(dict(data=name, b=b, rep=rep, method=m, tvd3=s.tvd3(syn, truth), viol=s.viol(syn, tabs, lo, hi),
                                 suppressed_share=float(np.mean(np.concatenate([x.ravel() for _, x in pub])))))
        print(name, "b", b, flush=True)
r = pd.DataFrame(rows)
r.to_csv(Path(args.out_dir) / "e02_rounding_m2.csv", index=False)
print(r.groupby(["data", "b", "method"])[["tvd3", "viol", "suppressed_share"]].mean().round(3).to_string())
