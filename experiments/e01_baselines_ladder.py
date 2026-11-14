"""E01 - S1/S2/S3 table ladder with classic generators (IPF + three integerisations, and a SASM-like annealer).

Aggregates come from 80% of the rows; synthetic data is scored on the held-out 20% (train-synthetic / test-real, TSTR)
and on the error of ALL 3-way margins (none of which is fitted). 'oracle_resample' = a fresh sample from the true joint
(the sampling-noise floor); 'real_train(TRTR)' = a model trained on the real training rows.
Output: results/e01_baselines_ladder.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from aggsyn import TableSpace, simulated_annealing, datasets
from aggsyn.cli import make_parser, setup

args = setup(make_parser(__doc__, default_datasets=["lanark"], default_reps=10).parse_args())
rows = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir)
    s = TableSpace(d)
    for rep, (tr, te) in enumerate(StratifiedShuffleSplit(args.reps, test_size=0.2, random_state=0).split(d, d.y)):
        dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
        N = len(dtr); truth = s.table(dtr); rng = np.random.default_rng(rep)
        a, p10 = s.tstr(dtr, dte)
        rows.append(dict(data=name, rep=rep, scen="-", method="real_train(TRTR)", tvd3=0, auc=a, p10=p10, fit_err=0))
        o = s.mc(truth / truth.sum(), N, rng); a, p10 = s.tstr(o, dte)
        rows.append(dict(data=name, rep=rep, scen="-", method="oracle_resample", tvd3=s.tvd3(o, truth), auc=a, p10=p10, fit_err=np.nan))
        for sc in ["S1", "S2", "S3"]:
            tabs = s.tabs(sc)
            tg = [s.marg(truth, ax) for ax in tabs]
            lo, hi = s.exact_intervals(truth, tabs)
            p = s.ipf(tabs, s.flat_margins(truth, tabs), N)
            outs = {"IPF+MC": s.mc(p, N, rng), "IPF+TRS": s.trs(p, N, rng),
                    "IPF+ILPround": s.ilp(p, N, tabs, lo, hi, interval=False, tl=3 if args.quick else 15),
                    "SA(SASM-like)": simulated_annealing(s, tabs, tg, N, rng, moves=2000 if args.quick else 20000)}
            for m, syn in outs.items():
                a, p10 = s.tstr(syn, dte)
                fit = np.mean([0.5 * np.abs(s.marg(s.table(syn), ax) - t).sum() / N for ax, t in zip(tabs, tg)])
                rows.append(dict(data=name, rep=rep, scen=sc, method=m, tvd3=s.tvd3(syn, truth), auc=a, p10=p10, fit_err=fit))
        print(name, "rep", rep, flush=True)
pd.DataFrame(rows).to_csv(Path(args.out_dir) / "e01_baselines_ladder.csv", index=False)
print(pd.DataFrame(rows).groupby(["scen", "method"])[["fit_err", "tvd3", "auc", "p10"]].mean().round(3).to_string())
