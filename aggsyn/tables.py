"""Core: contingency-table space, IPF / max-entropy fitting, interval (M2) fitting, integer rounding, metrics.

Variables are integer-coded columns of a DataFrame. The LAST column is the binary outcome ``y``.
Everything works on the full joint table (a dense numpy array), which is fine for the small
domains used in the paper (about 250 to 3500 cells).
"""
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
from scipy.sparse import csr_matrix, hstack, identity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class TableSpace:
    def __init__(self, df: pd.DataFrame):
        self.cols = list(df.columns)
        self.K = len(self.cols)
        self.L = self.K - 1                      # axis of the outcome
        self.dims = tuple(int(df[c].max()) + 1 for c in self.cols)
        self.idx = np.indices(self.dims).reshape(self.K, -1)

    # ---------- basic tables ----------
    def table(self, d: pd.DataFrame) -> np.ndarray:
        t = np.zeros(self.dims)
        np.add.at(t, tuple(d[c].values for c in self.cols), 1)
        return t

    def marg(self, t: np.ndarray, ax) -> np.ndarray:
        return t.sum(axis=tuple(i for i in range(self.K) if i not in ax))

    def tabs(self, scenario: str):
        """S1 = single-variable totals; S2 = totals + outcome x feature; S3 = totals + all pairs."""
        one = [(i,) for i in range(self.K)]
        if scenario == "S1":
            return one
        if scenario == "S2":
            return one + [(i, self.L) for i in range(self.L)]
        if scenario == "S3":
            return one + list(itertools.combinations(range(self.K), 2))
        raise ValueError(scenario)

    def A(self, tabs) -> csr_matrix:
        """Sparse 0/1 matrix mapping the flattened joint table to the stacked flattened margins."""
        rows, off = [], 0
        for ax in tabs:
            sh = [self.dims[i] for i in ax]
            rows.append(off + np.ravel_multi_index(tuple(self.idx[i] for i in ax), sh))
            off += int(np.prod(sh))
        r = np.concatenate(rows)
        c = np.tile(np.arange(self.idx.shape[1]), len(tabs))
        return csr_matrix((np.ones(len(r)), (r, c)), shape=(off, self.idx.shape[1]))

    def flat_margins(self, t, tabs) -> np.ndarray:
        return np.concatenate([self.marg(t, ax).ravel() for ax in tabs])

    # ---------- publishing (rounding / suppression) ----------
    def publish(self, truth, tabs, b=5):
        """Canadian-style release: true count < b -> suppressed, else round to nearest b."""
        out = []
        for ax in tabs:
            t = self.marg(truth, ax)
            out.append((b * np.round(t / b), t < b))
        return out

    def intervals(self, pub, b=5):
        """Ranges consistent with the release: suppressed -> [0, b-1]; rounded r -> [r-b/2, r+b/2]."""
        lo, hi = [], []
        for r, sup in pub:
            lo.append(np.where(sup, 0, np.maximum(r - b / 2, 0)).ravel())
            hi.append(np.where(sup, b - 1, r + b / 2).ravel())
        return np.concatenate(lo), np.concatenate(hi)

    def naive_targets(self, pub, suppressed_value=2.0):
        """What a standard analyst does: treat rounded numbers as exact, suppressed cells as ~2."""
        return np.concatenate([np.where(s, suppressed_value, r).ravel() for r, s in pub])

    def exact_intervals(self, truth, tabs):
        v = self.flat_margins(truth, tabs)
        return v.copy(), v.copy()

    # ---------- fitting ----------
    def ipf(self, tabs, target_flat, N, iters=300):
        """Iterative proportional fitting from a uniform seed = maximum-entropy joint matching the margins."""
        p = np.ones(self.dims) / np.prod(self.dims)
        off, tg = 0, []
        for ax in tabs:
            sh = [self.dims[i] for i in ax]
            n = int(np.prod(sh))
            tg.append(target_flat[off:off + n].reshape(sh) / N)
            off += n
        for _ in range(iters):
            for ax, t in zip(tabs, tg):
                cur = self.marg(p, ax)
                r = np.where(cur > 0, t / np.maximum(cur, 1e-300), 0)
                p = p * r.reshape([self.dims[i] if i in ax else 1 for i in range(self.K)])
        return p / p.sum()

    def m2(self, tabs, lo, hi, N):
        """Interval-constrained max entropy (method "M2"): max H(p) s.t. lo <= N*A p <= hi.
        Solved through its smooth dual with L-BFGS-B (u, v >= 0 are the multipliers of the upper/lower bounds)."""
        A = self.A(tabs)
        m = len(lo)

        def f(x):
            u, v = x[:m], x[m:]
            eta = A.T @ (v - u)
            mx = eta.max()
            w = np.exp(eta - mx)
            Z = w.sum()
            p = w / Z
            val = mx + np.log(Z) + (u @ hi - v @ lo) / N
            Ap = A @ p
            return N * val, N * np.r_[-Ap + hi / N, Ap - lo / N]

        r = minimize(f, np.zeros(2 * m), jac=True, method="L-BFGS-B", bounds=[(0, None)] * (2 * m),
                     options={"maxiter": 20000, "maxfun": 40000, "ftol": 1e-16, "gtol": 1e-9})
        eta = A.T @ (r.x[m:] - r.x[:m])
        p = np.exp(eta - eta.max())
        return (p / p.sum()).reshape(self.dims)

    # ---------- integerisation ----------
    def df(self, counts: np.ndarray) -> pd.DataFrame:
        ix = np.array(np.nonzero(counts)).T
        rep = counts[tuple(ix.T)].astype(int)
        return pd.DataFrame(np.repeat(ix, rep, axis=0), columns=self.cols)

    def mc(self, p, N, rng):
        """Monte Carlo (multinomial) sampling."""
        return self.df(rng.multinomial(N, p.ravel()).reshape(self.dims).astype(float))

    def trs(self, p, N, rng):
        """Truncate-replicate-sample integerisation."""
        e = p * N
        fl = np.floor(e)
        need = int(N - fl.sum())
        fr = (e - fl).ravel()
        if need > 0:
            f = fl.ravel()
            f[rng.choice(fr.size, need, replace=False, p=fr / fr.sum())] += 1
            fl = f.reshape(self.dims)
        return self.df(fl)

    def ilp(self, p, N, tabs, lo, hi, interval=False, tl=5):
        """Controlled rounding: each cell goes to floor or ceil of N*p, total = N.
        interval=False : minimise L1 distance of the margins to the midpoint of [lo, hi] (standard practice).
        interval=True  : minimise only the amount by which margins leave [lo, hi] (range-respecting rounding)."""
        e = (p * N).ravel()
        fl = np.floor(e)
        free = np.where(np.ceil(e) > fl)[0]
        need = int(round(N - fl.sum()))
        A = self.A(tabs)
        Af = A[:, free]
        base = A @ fl
        nf, m = len(free), len(lo)
        I = identity(m, format="csr")
        if interval:
            cons = [LinearConstraint(hstack([-Af, -I]), -np.inf, base - lo),
                    LinearConstraint(hstack([Af, -I]), -np.inf, hi - base)]
        else:
            t = (lo + hi) / 2
            cons = [LinearConstraint(hstack([Af, -I]), -np.inf, t - base),
                    LinearConstraint(hstack([-Af, -I]), -np.inf, base - t)]
        cons.append(LinearConstraint(np.r_[np.ones(nf), np.zeros(m)][None, :], need, need))
        res = milp(np.r_[np.zeros(nf), np.ones(m)], constraints=cons,
                   integrality=np.r_[np.ones(nf), np.zeros(m)],
                   bounds=Bounds(np.zeros(nf + m), np.r_[np.ones(nf), np.full(m, np.inf)]),
                   options={"time_limit": tl})
        z = fl.copy()
        if res.x is not None:
            z[free] += np.round(res.x[:nf])
        else:                                     # solver timed out without a solution: arbitrary fallback
            z[free[:need]] += 1
        return self.df(z.reshape(self.dims))

    # ---------- metrics ----------
    def viol(self, syn, tabs, lo, hi) -> float:
        """Share of published cells the synthetic data contradicts (falls outside its range)."""
        c = self.A(tabs) @ self.table(syn).ravel()
        return float(np.mean((c < lo - 1e-9) | (c > hi + 1e-9)))

    def tvd3(self, syn, truth) -> float:
        """Mean total-variation distance over ALL 3-way margins (none of which is fitted in S1-S3)."""
        a = self.table(syn) / len(syn)
        b = truth / truth.sum()
        v = [0.5 * np.abs(self.marg(a, ax) - self.marg(b, ax)).sum()
             for ax in itertools.combinations(range(self.K), 3)]
        return float(np.mean(v))

    def feats(self, d) -> np.ndarray:
        return np.hstack([np.eye(self.dims[i])[d[c].values] for i, c in enumerate(self.cols[:-1])])

    def tstr(self, syn, test):
        """Train on synthetic, test on real: (AUC, precision at top 10%) of a logistic regression."""
        y_col = self.cols[-1]
        if syn[y_col].nunique() < 2:
            return 0.5, 0.0
        m = LogisticRegression(max_iter=3000).fit(self.feats(syn), syn[y_col])
        sc = m.predict_proba(self.feats(test))[:, 1]
        y = test[y_col].values
        k = max(1, int(round(0.1 * len(test))))
        return roc_auc_score(y, sc), y[np.argsort(-sc, kind="stable")[:k]].mean()

    def uniq_repro(self, syn, truth) -> float:
        """Privacy proxy: share of people who are unique in the real data whose exact profile appears in the synthetic data."""
        u = truth == 1
        return float(((self.table(syn) >= 1) & u).sum() / max(u.sum(), 1))


G = TableSpace  # short alias used in the experiment scripts


def simulated_annealing(space: TableSpace, tabs, targets, N, rng, moves=20000, pool=20000):
    """SASM-style baseline: swap individuals from a candidate pool to reduce total absolute error on the margins.
    No seed survey is available, so candidates are drawn from the single-variable margins (independence).
    ``targets`` is a list of marginal COUNT arrays, one per table in ``tabs`` (the single-variable tables must be included)."""
    K, dims = space.K, space.dims
    p1 = {ax[0]: tg / tg.sum() for ax, tg in zip(tabs, targets) if len(ax) == 1}
    P = np.stack([rng.choice(dims[i], size=pool, p=p1[i]) for i in range(K)])
    cur = np.stack([rng.choice(dims[i], size=N, p=p1[i]) for i in range(K)])

    def flat(arr, ax):
        return np.ravel_multi_index(tuple(arr[i] for i in ax), [dims[i] for i in ax])

    PF = [flat(P, ax) for ax in tabs]
    CF = [flat(cur, ax) for ax in tabs]
    M = [np.bincount(cf, minlength=int(np.prod([dims[i] for i in ax]))).astype(float) for cf, ax in zip(CF, tabs)]
    T = [t.ravel() for t in targets]
    err = sum(np.abs(m - t).sum() for m, t in zip(M, T))
    T0 = max(err * 0.02, 1.0)
    for k in range(moves):
        temp = T0 * (1 - k / moves) ** 2 + 1e-6
        i, j = rng.integers(N), rng.integers(pool)
        dlt = 0.0
        for t in range(len(tabs)):
            o, n = CF[t][i], PF[t][j]
            if o != n:
                dlt += (abs(M[t][o] - 1 - T[t][o]) - abs(M[t][o] - T[t][o])
                        + abs(M[t][n] + 1 - T[t][n]) - abs(M[t][n] - T[t][n]))
        if dlt <= 0 or rng.random() < np.exp(-dlt / temp):
            for t in range(len(tabs)):
                o, n = CF[t][i], PF[t][j]
                M[t][o] -= 1
                M[t][n] += 1
                CF[t][i] = n
            cur[:, i] = P[:, j]
    return pd.DataFrame(cur.T, columns=space.cols)
