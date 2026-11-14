"""E08 - utility vs disclosure risk: how much usefulness survives a small-cell budget?

Risk of a table = number of cells holding 1-4 people (the cells agencies normally must hide). A budget R caps the total.
Strategies: plain task-aware greedy (add tables in order while they fit), risk-aware greedy (gain per 1 + risk),
label-first, random-within-budget (NOTE: random uses many more tables than the others, so it is not a fair comparison).
Utility = AUC loss of a model trained on the synthetic people vs one trained on the real rows.
Output: results/e08_privacy_risk_frontier.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from aggsyn import TableSpace, Selector, datasets
from aggsyn.metrics import mutual_information
from aggsyn.cli import make_parser, setup

args = setup(make_parser(__doc__, default_reps=5).parse_args())
budgets = (0, 6) if args.quick else (0, 2, 6, 12, 24)
res = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir); s = TableSpace(d); S = Selector(s, tol=1e-7, maxit=300)
    L, pairs, ones = s.L, S.pairs, S.ones; ntr = datasets.TRAIN_SIZES[name]
    for rep in range(args.reps):
        S.clear(); rng = np.random.default_rng(rep)
        tr, te = next(StratifiedShuffleSplit(1, train_size=ntr, test_size=min(3000, len(d) - ntr), random_state=rep).split(d, d.y))
        dtr = d.iloc[tr].reset_index(drop=True); dte = d.iloc[te].reset_index(drop=True); N = len(dtr); t = s.table(dtr); p = t / N
        Xte, yte = s.feats(dte), dte.y.values
        trtr = roc_auc_score(yte, LogisticRegression(max_iter=3000).fit(s.feats(dtr), dtr.y.values).predict_proba(Xte)[:, 1])
        risk = {e: int(((m := s.marg(t, e)) >= 1).__and__(m < 5).sum()) for e in pairs}
        base = S.loss(p, [], "cond")
        gain = lambda sel: base - S.loss(p, sel, "cond")
        order = S.greedy(p, 6, "cond")
        MIv = {e: mutual_information(s, p, *e) for e in pairs}
        lab = sorted([(i, L) for i in range(L)], key=lambda e: -MIv[e])

        def within(seq, R):
            sel, c = [], 0
            for e in seq:
                if c + risk[e] <= R:
                    sel.append(e); c += risk[e]
            return sel

        def riskaware(R):
            sel, c = [], 0
            while len(sel) < 6:
                best = None
                for e in pairs:
                    if e in sel or c + risk[e] > R:
                        continue
                    gn = gain(sel + [e]) - (gain(sel) if sel else 0)
                    sc = gn / (1 + risk[e])
                    if best is None or sc > best[0]:
                        best = (sc, e, gn)
                if best is None or best[2] < 1e-4:
                    break
                sel.append(best[1]); c += risk[best[1]]
            return sel

        for R in budgets:
            rr = np.random.default_rng(100 + rep); rnd = [pairs[i] for i in rr.permutation(len(pairs))]
            for nm, sel in [("Plain greedy (ignore risk, stop at limit)", within(order, R)), ("Risk-aware greedy", riskaware(R)),
                            ("Label-first", within(lab, R)), ("Random", within(rnd, R))]:
                tabs = ones + list(sel); pub = s.publish(t, tabs, 5)
                q = s.ipf(tabs, s.naive_targets(pub), N, iters=200); syn = s.trs(q, N, rng)
                a = 0.5 if syn.y.nunique() < 2 else roc_auc_score(yte, LogisticRegression(max_iter=3000).fit(s.feats(syn), syn.y.values).predict_proba(Xte)[:, 1])
                res.append(dict(data=name, rep=rep, R=R, strategy=nm, k=len(sel), risk_used=sum(risk[e] for e in sel), auc_loss=trtr - a))
        print(name, "rep", rep, flush=True)
r = pd.DataFrame(res)
r.to_csv(Path(args.out_dir) / "e08_privacy_risk_frontier.csv", index=False)
print(r.pivot_table(index="strategy", columns="R", values="auc_loss").round(3).to_string())
