"""E10 - the pre-release AUC-loss certificate (Theorem 4 and its sharpened form), checked against the exact loss.

For several table sets, compare the exact AUC loss of the scorer implied by the tables with
 - certificate   (new, computable from microdata before publishing)
 - bound_tv, bound_pinsker (older, looser).
NOTE: the loss is measured against the best possible scorer on the same features; on small datasets (Lanark, Mroz) that 'best' scorer is
over-fitted to a few hundred rows, so use Adult / GSS / GSS34 for tight readings. The scorer is the tables' own posterior q(y=1|x), not a
retrained model (which loses less).
Output: results/e10_auc_certificate.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from aggsyn import TableSpace, Selector, datasets, population_certificate
from aggsyn.cli import make_parser, setup

args = setup(make_parser(__doc__).parse_args())
rows = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir); s = TableSpace(d); S = Selector(s, tol=1e-10, maxit=1500)
    t = s.table(d); p = t / t.sum(); L = s.L
    sets = {"S1 (totals only)": [], "S2 (outcome x feature)": [(i, L) for i in range(L)],
            "Task-aware k=3": S.greedy(p, 3, "cond"), "Task-aware k=6": S.greedy(p, min(6, len(S.pairs)), "cond"),
            "S3 (all pairs)": S.pairs}
    if args.quick:
        sets = {k: v for k, v in sets.items() if k in ("S1 (totals only)", "S2 (outcome x feature)")}
    for nm, sel in sets.items():
        q = S.fit(p, S.ones + sel)
        c = population_certificate(s, p, q)
        rows.append(dict(data=name, tables=nm, regret=c["regret"], certificate=c["certificate"], bound_tv=c["bound_tv"],
                         bound_pinsker=c["bound_pinsker"], valid=c["regret"] <= c["certificate"] + 1e-9))
    print(name, "done", flush=True)
r = pd.DataFrame(rows)
r.to_csv(Path(args.out_dir) / "e10_auc_certificate.csv", index=False)
print("certificate valid in", int(r.valid.sum()), "of", len(r))
print(r.groupby("tables")[["regret", "certificate", "bound_tv", "bound_pinsker"]].mean().round(3).to_string())
