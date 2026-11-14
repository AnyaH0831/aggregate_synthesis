"""E09 - is greedy table selection provably good? Exhaustive check on exact tables (all sets of up to 3 pair-tables).

Utility u(S) = (outcome-given-features error with no pair tables) - (error with the pair tables S).
Reports: greedy / optimum ratio for budgets 1..3; share of checks where diminishing returns (submodularity) fails;
share of cases where ADDING a table INCREASES the task error (task error is not monotone); the submodularity ratio gamma;
and the a-priori bound for outcome x feature tables via the conditional total correlation TC(X|Y) = KL(p||q_S2).
Output: results/e09_greedy_submodularity.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import itertools
import numpy as np
import pandas as pd
from aggsyn import TableSpace, Selector, datasets, metrics
from aggsyn.cli import make_parser, setup

args = setup(make_parser(__doc__).parse_args())
rows = []
for name in args.datasets:
    d = datasets.load(name, args.data_dir); s = TableSpace(d); S = Selector(s, tol=1e-8, maxit=400)
    pairs, ones, L = S.pairs, S.ones, s.L
    t = s.table(d); p = t / t.sum(); base = S.loss(p, [], "cond")
    kmax = 2 if args.quick else 3
    u = {(): 0.0}
    for k in range(1, kmax + 1):
        for T in itertools.combinations(pairs, k):
            u[T] = base - S.loss(p, list(T), "cond")
    opt = {k: max(v for T, v in u.items() if len(T) == k) for k in range(1, kmax + 1)}
    gr = S.greedy(p, kmax, "cond")
    ratio = {k: u[tuple(sorted(gr[:k]))] / opt[k] for k in range(1, kmax + 1)}
    viol = tot = 0
    for T in [()] + [(a,) for a in pairs]:
        for f_, e_ in itertools.permutations([x for x in pairs if x not in T], 2):
            Tf, Te, Tfe = tuple(sorted(T + (f_,))), tuple(sorted(T + (e_,))), tuple(sorted(T + (f_, e_)))
            if len(Tfe) > kmax:
                continue
            tot += 1
            viol += (u[Tfe] - u[Tf]) > (u[Te] - u[T]) + 1e-9
    dec = mon = 0
    for T, v in u.items():
        if len(T) >= kmax:
            continue
        for e in pairs:
            if e in T:
                continue
            mon += 1
            dec += (u[tuple(sorted(T + (e,)))] - v) < -1e-9
    gam = []
    for T in [()] + [(a,) for a in pairs]:
        for a, b in itertools.combinations([x for x in pairs if x not in T], 2):
            Tab = tuple(sorted(T + (a, b)))
            if len(Tab) > kmax:
                continue
            den = u[Tab] - u[T]
            if den > 1e-6:
                gam.append(((u[tuple(sorted(T + (a,)))] - u[T]) + (u[tuple(sorted(T + (b,)))] - u[T])) / den)
    q2 = S.fit(p, ones + [(i, L) for i in range(L)]); tc = metrics.kl(p, q2); pi = p[..., 1].sum()
    rows.append(dict(data=name, **{f"greedy_over_opt_k{k}": ratio[k] for k in ratio}, submod_violations=viol / max(tot, 1),
                     adding_table_hurts=dec / max(mon, 1), gamma_p5=float(np.percentile(gam, 5)) if gam else np.nan,
                     gamma_min=min(gam) if gam else np.nan, cond_total_corr=tc, condKL_S2=metrics.cond_kl(p, q2),
                     pinsker_bound_S2=float(np.sqrt(tc / 2) / (pi * (1 - pi)))))
    print(name, "done", flush=True)
r = pd.DataFrame(rows)
r.to_csv(Path(args.out_dir) / "e09_greedy_submodularity.csv", index=False)
print(r.round(3).to_string(index=False))
