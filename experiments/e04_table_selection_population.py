"""E04 - which pair-tables to publish? Population-level comparison (exact tables, no sampling noise).

Budget k = number of pair tables beyond the always-published single-variable totals.
Strategies: task-aware greedy (min outcome-given-features error), task-agnostic greedy (min joint KL),
MI-tree (Chow-Liu / MST-like), label-first (outcome x feature ranked by MI), random (20 draws).
Metrics: KL, conditional KL, and the AUC (on the full population) of a logistic regression trained on the implied joint.
Output: results/e04_table_selection_population.csv  (rows with strategy '_randdraw' are the individual random draws)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from aggsyn import TableSpace, Selector, mi_tree, label_first, random_pairs, datasets, metrics
from aggsyn.cli import make_parser, setup

p = make_parser(__doc__, default_reps=20)
p.add_argument("--kmax", type=int, default=8)
args = setup(p.parse_args())
kmax = 3 if args.quick else args.kmax
draws = 3 if args.quick else args.reps
res = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir); s = TableSpace(d)
    S = Selector(s, tol=1e-10, maxit=1500)
    t = s.table(d); pop = t / t.sum(); base_auc = metrics.population_lr_auc(s, pop, pop)
    def ev(sel):
        q = S.fit(pop, S.ones + list(sel))
        return dict(kl=metrics.kl(pop, q), ckl=metrics.cond_kl(pop, q), auc=metrics.population_lr_auc(s, pop, q))
    seqs = {"KL-greedy (task-agnostic)": S.greedy(pop, kmax, "joint"), "LogLoss-greedy (task-aware)": S.greedy(pop, kmax, "cond"),
            "MI-tree (Chow-Liu)": mi_tree(s, pop, kmax), "Label-first (outcome x feature)": label_first(s, pop, kmax)}
    rng = np.random.default_rng(0)
    for k in range(1, kmax + 1):
        for nm, seq in seqs.items():
            res.append(dict(data=name, k=k, strategy=nm, **ev(seq[:k]), base_auc=base_auc))
        rs = [ev(random_pairs(s, k, rng)) for _ in range(draws)]
        res.append(dict(data=name, k=k, strategy="Random (mean of draws)", **{m: np.mean([r[m] for r in rs]) for m in ("kl", "ckl", "auc")}, base_auc=base_auc))
        res += [dict(data=name, k=k, strategy="_randdraw", **r, base_auc=base_auc) for r in rs]
    print(name, "done", flush=True)
r = pd.DataFrame(res)
r.to_csv(Path(args.out_dir) / "e04_table_selection_population.csv", index=False)
x = r[r.strategy != "_randdraw"].assign(auc_loss=lambda z: z.base_auc - z.auc)
print(x.pivot_table(index="k", columns="strategy", values="auc_loss").round(3).to_string())
