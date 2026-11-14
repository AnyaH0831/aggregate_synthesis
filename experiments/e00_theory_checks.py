"""E00 - numerical checks of the theory (Results 1-5), on exact tables (no sampling noise).

 (a) ladder      : S1/S2/S3 tables -> KL(p||q), entropy gap H(q)-H(p), TV, AUC of a model trained on q vs on p
 (b) naive Bayes : max-entropy joint for S2 equals naive Bayes (max abs difference)
 (c) separation  : toy example where joint-KL-greedy picks the wrong table (Result 5)
Outputs: results/e00_ladder.csv, e00_naive_bayes.csv, e00_separation.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from aggsyn import TableSpace, Selector, datasets, metrics
from aggsyn.cli import make_parser, setup

args = setup(make_parser(__doc__).parse_args())
ladder, nb_rows = [], []
for name in args.datasets:
    d = datasets.load(name, args.data_dir)
    s = TableSpace(d); S = Selector(s, tol=1e-11, maxit=2000)
    t = s.table(d); p = t / t.sum(); L = s.L
    base = metrics.population_lr_auc(s, p, p)
    for sc in ["S1", "S2", "S3"]:
        q = S.fit(p, s.tabs(sc))
        ladder.append(dict(data=name, tables=sc, KL=metrics.kl(p, q), entropy_gap=metrics.entropy(q) - metrics.entropy(p),
                           TV=0.5 * np.abs(p - q).sum(), cond_KL=metrics.cond_kl(p, q),
                           auc_train_on_q=metrics.population_lr_auc(s, p, q), auc_train_on_p=base))
    # naive Bayes equivalence
    py = s.marg(p, (L,))
    nb = np.zeros(s.dims)
    for yv in range(s.dims[L]):
        sl = np.ones(s.dims[:L])
        for i in range(L):
            m = s.marg(p, (i, L))[:, yv] / py[yv]
            sl = sl * m.reshape([s.dims[j] if j == i else 1 for j in range(L)])
        nb[..., yv] = sl * py[yv]
    q2 = S.fit(p, s.tabs("S2"))
    nb_rows.append(dict(data=name, max_abs_diff_maxent_vs_naive_bayes=float(np.abs(q2 - nb).max())))
    print(name, "done", flush=True)
pd.DataFrame(ladder).to_csv(Path(args.out_dir) / "e00_ladder.csv", index=False)
pd.DataFrame(nb_rows).to_csv(Path(args.out_dir) / "e00_naive_bayes.csv", index=False)

# (c) separation example: Y = noisy copy of X1 (90%); X2 = X3 uniform on m values, independent of (X1, Y).
sep = []
for m in (2, 4, 8, 16):
    P = np.zeros((2, m, m, 2))
    for x1 in range(2):
        for x2 in range(m):
            for y in range(2):
                P[x1, x2, x2, y] = 0.5 * (1 / m) * (0.9 if y == x1 else 0.1)
    df = pd.DataFrame([[v - 1 for v in P.shape]], columns=["x1", "x2", "x3", "y"])
    s = TableSpace(df); S = Selector(s, tol=1e-12, maxit=3000)
    for nm, pr in [("pair(X2,X3)  [what joint-KL-greedy picks]", (1, 2)), ("pair(X1,Y)   [task-aware]", (0, 3))]:
        q = S.fit(P, S.ones + [pr])
        etaq = q / np.maximum(q.sum(-1, keepdims=True), 1e-300)
        sep.append(dict(m=m, table=nm, joint_KL_left=metrics.kl(P, q), cond_KL_left=metrics.cond_kl(P, q),
                        auc_of_implied_posterior=metrics.weighted_auc(P, etaq.reshape(-1, 2)[:, 1])))
pd.DataFrame(sep).to_csv(Path(args.out_dir) / "e00_separation.csv", index=False)
print(pd.DataFrame(sep).round(3).to_string(index=False))
