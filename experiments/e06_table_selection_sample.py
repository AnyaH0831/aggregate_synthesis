"""E06 - the realistic version of E04: n ~ 500 people, tables ROUNDED to 5 with cells < 5 HIDDEN, 15 repeats, two classifiers.

The data holder chooses tables on its own sample, publishes them rounded, and the user fits IPF on the published numbers,
integerises, and trains a classifier on the synthetic people. Scored on held-out real people.
'auc_direct' scores people straight from the tables' implied posterior q(y=1|x) (no synthetic data, no classifier).
AUC LOSS is measured against a model trained on the real training rows (the reference is averaged per dataset).
Output: results/e06_table_selection_sample.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from aggsyn import TableSpace, Selector, mi_tree, label_first, random_pairs, datasets
from aggsyn.cli import make_parser, setup

args = setup(make_parser(__doc__, default_reps=15).parse_args())


def lr_auc(Xtr, ytr, Xte, yte):
    return 0.5 if len(set(ytr)) < 2 else roc_auc_score(yte, LogisticRegression(max_iter=3000).fit(Xtr, ytr).predict_proba(Xte)[:, 1])


def xgb_auc(Xtr, ytr, Xte, yte):
    if len(set(ytr)) < 2:
        return 0.5
    m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0, n_jobs=1).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


rows = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir); s = TableSpace(d); S = Selector(s, tol=1e-8, maxit=400)
    K, L = s.K, s.L; ntr = datasets.TRAIN_SIZES[name]; pairs = S.pairs
    for rep in range(args.reps):
        S.clear(); rng = np.random.default_rng(rep)
        tr, te = next(StratifiedShuffleSplit(1, train_size=ntr, test_size=min(3000, len(d) - ntr), random_state=rep).split(d, d.y))
        dtr = d.iloc[tr].reset_index(drop=True); dte = d.iloc[te].reset_index(drop=True); N = len(dtr)
        t = s.table(dtr); p = t / N
        Xte, yte, Xtr = s.feats(dte), dte.y.values, s.feats(dtr)
        rows.append(dict(data=name, rep=rep, strategy="real_train(TRTR)", k=0, auc_lr=lr_auc(Xtr, dtr.y.values, Xte, yte),
                         auc_xgb=xgb_auc(Xtr, dtr.y.values, Xte, yte), auc_direct=np.nan))
        cfgs = [("Task-aware greedy", 3, S.greedy(p, 3, "cond")), ("KL-greedy", 3, S.greedy(p, 3, "joint")),
                ("MI-tree (MST/Chow-Liu)", 3, mi_tree(s, p, 3)), ("Label-first", 3, label_first(s, p, 3)),
                ("Random", 3, random_pairs(s, 3, rng)),
                ("MI-tree (MST/Chow-Liu)", 6, mi_tree(s, p, 6)), ("Label-first", min(6, L), label_first(s, p, min(6, L))),
                ("Random", 6, random_pairs(s, 6, rng)),
                ("All outcome x feature (S2)", L, [(i, L) for i in range(L)]), ("All pairs (S3)", len(pairs), pairs)]
        if args.quick:
            cfgs = cfgs[:5]
        for nm, k, sel in cfgs:
            tabs = S.ones + list(sel)
            pub = s.publish(t, tabs, 5)
            q = s.ipf(tabs, s.naive_targets(pub), N, iters=250)
            syn = s.trs(q, N, rng)
            pos = q[..., 1] / np.maximum(q.sum(-1), 1e-300)
            sc = pos[tuple(dte[c].values for c in s.cols[:-1])]
            rows.append(dict(data=name, rep=rep, strategy=nm, k=k, auc_lr=lr_auc(s.feats(syn), syn.y.values, Xte, yte),
                             auc_xgb=xgb_auc(s.feats(syn), syn.y.values, Xte, yte), auc_direct=roc_auc_score(yte, sc)))
        print(name, "rep", rep, flush=True)
r = pd.DataFrame(rows)
r.to_csv(Path(args.out_dir) / "e06_table_selection_sample.csv", index=False)
ref = r[r.strategy == "real_train(TRTR)"].groupby("data")[["auc_lr", "auc_xgb"]].mean()
x = r[r.strategy != "real_train(TRTR)"].copy()
x["lr_loss"] = x.data.map(ref.auc_lr) - x.auc_lr
x["xgb_loss"] = x.data.map(ref.auc_xgb) - x.auc_xgb
x["direct_loss"] = x.data.map(ref.auc_lr) - x.auc_direct
print(x.groupby(["strategy", "k"])[["lr_loss", "xgb_loss", "direct_loss"]].mean().round(3).to_string())
