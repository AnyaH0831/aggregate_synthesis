"""E03 - four method variants on all datasets, all-pairs tables rounded to 5 and 10 with small cells hidden.

 IPF-naive+midILP      standard practice
 IPF-naive+intervalILP range-respecting rounding (keeps every published range)
 M2+midILP, M2+intervalILP   interval-fitted joint, then either rounding
Measures: error on held-out 3-way margins, share of published ranges contradicted ('viol'), TSTR AUC, and
'uniq' = share of people unique in the real data who are reproduced exactly (privacy proxy; a perfect re-draw gives ~0.63-0.69).
Output: results/e03_rounding_all_datasets.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from aggsyn import TableSpace, datasets
from aggsyn.cli import make_parser, setup

args = setup(make_parser(__doc__, default_reps=5).parse_args())
rows = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir); s = TableSpace(d); ntr = datasets.TRAIN_SIZES[name]
    for rep in range(args.reps):
        rng = np.random.default_rng(rep)
        tr, te = next(StratifiedShuffleSplit(1, train_size=ntr, test_size=min(3000, len(d) - ntr), random_state=rep).split(d, d.y))
        dtr = d.iloc[tr].reset_index(drop=True); dte = d.iloc[te].reset_index(drop=True); N = len(dtr); truth = s.table(dtr)
        a, _ = s.tstr(dtr, dte)
        rows.append(dict(data=name, b=0, rep=rep, method="real_train(TRTR)", tvd3=0, auc=a, viol=0, uniq=1.0))
        o = s.mc(truth / truth.sum(), N, rng); a, _ = s.tstr(o, dte)
        rows.append(dict(data=name, b=0, rep=rep, method="oracle_resample", tvd3=s.tvd3(o, truth), auc=a, viol=0, uniq=s.uniq_repro(o, truth)))
        tabs = s.tabs("S3")
        for b in [5, 10]:
            pub = s.publish(truth, tabs, b); lo, hi = s.intervals(pub, b)
            pn = s.ipf(tabs, s.naive_targets(pub), N); pm = s.m2(tabs, lo, hi, N)
            for m, (pp, iv) in {"IPF-naive+midILP": (pn, False), "IPF-naive+intervalILP": (pn, True),
                                "M2+midILP": (pm, False), "M2+intervalILP": (pm, True)}.items():
                syn = s.ilp(pp, N, tabs, lo, hi, iv, tl=3 if args.quick else 6); a, _ = s.tstr(syn, dte)
                rows.append(dict(data=name, b=b, rep=rep, method=m, tvd3=s.tvd3(syn, truth), auc=a,
                                 viol=s.viol(syn, tabs, lo, hi), uniq=s.uniq_repro(syn, truth)))
        print(name, "rep", rep, flush=True)
r = pd.DataFrame(rows)
r.to_csv(Path(args.out_dir) / "e03_rounding_all_datasets.csv", index=False)
print(r[r.b > 0].groupby(["data", "b", "method"])[["tvd3", "viol", "auc", "uniq"]].mean().round(3).to_string())
