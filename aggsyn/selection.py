"""Which tables to publish: greedy (task-aware or task-agnostic), information tree, label-first, random.

All selection is done by the data holder, who has the microdata, on the empirical table ``p`` (a probability array).
A "table" below is a PAIR of variable indices, e.g. (2, 6) = variable 2 crossed with the outcome (index L).
Single-variable totals are always published and are not part of the budget.
"""
import itertools
import numpy as np
from .metrics import kl, cond_kl, mutual_information


class Selector:
    def __init__(self, space, tol=1e-8, maxit=400):
        self.s = space
        self.K, self.L = space.K, space.L
        self.pairs = list(itertools.combinations(range(self.K), 2))
        self.ones = [(i,) for i in range(self.K)]
        self.tol, self.maxit = tol, maxit
        self._cache = {}

    def clear(self):
        self._cache.clear()

    def fit(self, p, tabs):
        """Max-entropy joint matching the margins of ``p`` for the given tables (IPF from a uniform seed, cached)."""
        key = (id(p), tuple(sorted(tabs)))
        if key in self._cache:
            return self._cache[key]
        s = self.s
        tg = [s.marg(p, ax) for ax in key[1]]
        q = np.ones(s.dims) / np.prod(s.dims)
        for _ in range(self.maxit):
            dev = 0.0
            for ax, t in zip(key[1], tg):
                cur = s.marg(q, ax)
                dev = max(dev, np.abs(cur - t).max())
                r = np.where(cur > 0, t / np.maximum(cur, 1e-300), 0)
                q = q * r.reshape([s.dims[i] if i in ax else 1 for i in range(self.K)])
            if dev < self.tol:
                break
        self._cache[key] = q
        return q

    def loss(self, p, sel, criterion="cond"):
        q = self.fit(p, self.ones + list(sel))
        return cond_kl(p, q) if criterion == "cond" else kl(p, q)

    def greedy(self, p, k, criterion="cond"):
        """criterion='cond'  -> TASK-AWARE greedy: minimise the outcome-given-features error.
           criterion='joint' -> task-agnostic greedy: minimise the joint KL (keeps the most overall structure)."""
        sel = []
        for _ in range(k):
            best = None
            for pr in self.pairs:
                if pr in sel:
                    continue
                v = self.loss(p, sel + [pr], criterion)
                if best is None or v < best[0]:
                    best = (v, pr)
            sel.append(best[1])
        return sel


def mi_tree(space, p, k):
    """Chow-Liu / MST-style: maximum-spanning-tree on pairwise mutual information (noiseless analogue of MST)."""
    pairs = list(itertools.combinations(range(space.K), 2))
    MI = {e: mutual_information(space, p, *e) for e in pairs}
    order = sorted(pairs, key=lambda e: -MI[e])
    comp = list(range(space.K))

    def find(a):
        while comp[a] != a:
            a = comp[a]
        return a

    sel = []
    for e in order:
        if find(e[0]) != find(e[1]):
            comp[find(e[0])] = find(e[1])
            sel.append(e)
    return (sel + [e for e in order if e not in sel])[:k]


def label_first(space, p, k):
    """Outcome x feature tables ranked by mutual information with the outcome, then the other pairs by MI."""
    L = space.L
    pairs = list(itertools.combinations(range(space.K), 2))
    MI = {e: mutual_information(space, p, *e) for e in pairs}
    lab = sorted([(i, L) for i in range(L)], key=lambda e: -MI[e])
    return (lab + [e for e in sorted(pairs, key=lambda e: -MI[e]) if e not in lab])[:k]


def random_pairs(space, k, rng):
    pairs = list(itertools.combinations(range(space.K), 2))
    return [pairs[i] for i in rng.choice(len(pairs), k, replace=False)]
