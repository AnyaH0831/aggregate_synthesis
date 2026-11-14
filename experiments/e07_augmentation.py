"""E07 - does synthetic data help when the real labelled sample is small?

Setting: a data holder publishes 3 task-aware pair-tables (rounded to 5) computed on a LARGE pool of people; you only have
n = 30..250 labelled people. Compare: real only | synthetic-from-tables only | real + synthetic (synthetic total weight = lam * n).
Caveat (important): the gain comes from the tables summarising far more people than the small sample; synthetic data is just the
vehicle that carries that information into a standard model. For Lanark and Mroz the pool is small, so the small sample is a subset of it.
Output: results/e07_augmentation.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from aggsyn import TableSpace, Selector, datasets
from aggsyn.cli import make_parser, setup

args = setup(make_parser(__doc__, default_reps=15).parse_args())
rows = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir); s = TableSpace(d); S = Selector(s)
    pool, test = train_test_split(d, test_size=0.3 if len(d) < 2000 else min(3000, len(d) // 3), stratify=d.y, random_state=0)
    if len(pool) > 20000:
        pool = pool.sample(20000, random_state=0)
    pool = pool.reset_index(drop=True); test = test.reset_index(drop=True); Np = len(pool)
    t = s.table(pool); p = t / Np
    sel = S.greedy(p, 3, "cond"); tabs = S.ones + sel; pub = s.publish(t, tabs, 5)
    q = s.ipf(tabs, s.naive_targets(pub), Np, iters=250)
    rng = np.random.default_rng(1); nsyn = min(Np, 5000); syn = s.trs(q, nsyn, rng)
    Xs, ys, Xte, yte = s.feats(syn), syn.y.values, s.feats(test), test.y.values

    def fit_lr(X, y, w=None):
        return LogisticRegression(max_iter=3000).fit(X, y, sample_weight=w)

    def fit_xgb(X, y, w=None):
        return xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0, n_jobs=1).fit(X, y, sample_weight=w)

    auc = lambda m: roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    syn_lr, syn_x, full = auc(fit_lr(Xs, ys)), auc(fit_xgb(Xs, ys)), auc(fit_lr(s.feats(pool), pool.y.values))
    print(name, "tables", sel, "synthetic-only", round(syn_lr, 3), "real-full", round(full, 3), flush=True)
    for n in ([30, 120] if args.quick else [30, 60, 120, 250]):
        if n >= Np:
            continue
        for rep in range(args.reps):
            r2 = np.random.default_rng(1000 * n + rep)
            while True:
                idx = r2.choice(Np, n, replace=False); ysm = pool.y.values[idx]
                if 3 <= ysm.sum() <= n - 3:
                    break
            Xr = s.feats(pool.iloc[idx])
            row = dict(data=name, n=n, rep=rep, real_lr=auc(fit_lr(Xr, ysm)), real_xgb=auc(fit_xgb(Xr, ysm)),
                       syn_lr=syn_lr, syn_xgb=syn_x, full_real_lr=full)
            for lam in (0.5, 1, 2, 4, 8):
                X = np.vstack([Xr, Xs]); y = np.r_[ysm, ys]; w = np.r_[np.ones(n), np.full(len(ys), lam * n / len(ys))]
                row[f"comb_lr_{lam}"] = auc(fit_lr(X, y, w))
                if lam in (1, 4):
                    row[f"comb_xgb_{lam}"] = auc(fit_xgb(X, y, w))
            rows.append(row)
r = pd.DataFrame(rows)
r.to_csv(Path(args.out_dir) / "e07_augmentation.csv", index=False)
print(r.groupby("n")[["real_lr", "syn_lr", "comb_lr_2", "real_xgb", "comb_xgb_4"]].mean().round(3).to_string())
