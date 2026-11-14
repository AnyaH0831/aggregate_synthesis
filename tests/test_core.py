"""Unit tests for the facts the paper relies on. Run:  python -m pytest -q"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import itertools
import numpy as np
import pandas as pd
import pytest
from aggsyn import TableSpace, Selector, population_certificate, subgroup_rate_bounds, metrics


def toy(n=3000, seed=0):
    """6 features + binary outcome with pairwise structure."""
    r = np.random.default_rng(seed)
    x1 = r.integers(0, 3, n); x2 = r.integers(0, 2, n); x3 = (x1 + r.integers(0, 2, n)) % 3
    y = (r.random(n) < 0.2 + 0.3 * (x1 == 2) + 0.2 * x2).astype(int)
    return pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, y=y))


def setup():
    d = toy(); s = TableSpace(d); t = s.table(d); return d, s, t / t.sum()


def test_ipf_matches_margins():
    d, s, p = setup(); tabs = s.tabs("S3")
    q = s.ipf(tabs, s.flat_margins(p, tabs), 1.0, iters=500)
    for ax in tabs:
        assert np.allclose(s.marg(q, ax), s.marg(p, ax), atol=1e-6)


def test_s2_maxent_is_naive_bayes():
    """Result 3: max-entropy joint for totals + outcome x feature tables is naive Bayes."""
    d, s, p = setup(); S = Selector(s, tol=1e-12, maxit=3000); L = s.L
    q = S.fit(p, s.tabs("S2")); py = s.marg(p, (L,)); nb = np.zeros(s.dims)
    for yv in range(2):
        sl = np.ones(s.dims[:L])
        for i in range(L):
            m = s.marg(p, (i, L))[:, yv] / py[yv]
            sl = sl * m.reshape([s.dims[j] if j == i else 1 for j in range(L)])
        nb[..., yv] = sl * py[yv]
    assert np.abs(q - nb).max() < 1e-9


def test_entropy_gap_identity():
    """Result 2: KL(p||q) = H(q) - H(p) for the max-entropy fit."""
    d, s, p = setup(); S = Selector(s, tol=1e-12, maxit=3000)
    q = S.fit(p, s.tabs("S2"))
    assert abs(metrics.kl(p, q) - (metrics.entropy(q) - metrics.entropy(p))) < 1e-8


def test_chain_rule_decomposition():
    """Result 1: KL(p||q) = KL(p_X||q_X) + conditional KL."""
    d, s, p = setup(); S = Selector(s, tol=1e-12, maxit=3000)
    q = S.fit(p, s.tabs("S2"))
    assert abs(metrics.kl(p, q) - (metrics.kl(p.sum(-1), q.sum(-1)) + metrics.cond_kl(p, q))) < 1e-8


def test_certificate_is_valid_and_ordered():
    """Theorem 4: exact regret <= certificate <= older TV bound."""
    d, s, p = setup(); S = Selector(s, tol=1e-10, maxit=1500)
    for sel in ([], [(0, s.L)], [(0, s.L), (1, s.L)], S.pairs):
        q = S.fit(p, S.ones + sel); c = population_certificate(s, p, q)
        assert c["regret"] <= c["certificate"] + 1e-9
        assert c["certificate"] <= c["bound_tv"] + 1e-9


def test_task_aware_beats_task_agnostic_on_separation_example():
    """Result 5: joint-KL greedy picks (X2,X3); task-aware picks (X1,Y)."""
    m = 8; P = np.zeros((2, m, m, 2))
    for x1 in range(2):
        for x2 in range(m):
            for y in range(2):
                P[x1, x2, x2, y] = 0.5 / m * (0.9 if y == x1 else 0.1)
    s = TableSpace(pd.DataFrame([[v - 1 for v in P.shape]], columns=["x1", "x2", "x3", "y"]))
    S = Selector(s, tol=1e-12, maxit=3000)
    assert S.greedy(P, 1, "joint")[0] == (1, 2)
    assert S.greedy(P, 1, "cond")[0] == (0, 3)
    assert metrics.cond_kl(P, S.fit(P, S.ones + [(0, 3)])) < 1e-9


def test_m2_satisfies_ranges_and_matches_ipf_when_exact():
    d, s, p = setup(); N = len(d); tabs = s.tabs("S2")
    lo = hi = s.flat_margins(p, tabs) * N
    q = s.m2(tabs, lo, hi, N)
    assert np.allclose(s.A(tabs) @ q.ravel() * N, lo, atol=1e-2)


def test_ilp_rounding_keeps_total_and_ranges():
    d, s, p = setup(); N = 400; tabs = s.tabs("S2"); t = s.table(d.sample(N, random_state=1))
    pub = s.publish(t, tabs, 5); lo, hi = s.intervals(pub, 5)
    q = s.m2(tabs, lo, hi, N); syn = s.ilp(q, N, tabs, lo, hi, interval=True, tl=10)
    assert len(syn) == N
    assert s.viol(syn, tabs, lo, hi) <= 0.05


def test_lp_bounds_contain_truth_and_are_wide_without_tables():
    d, s, p = setup(); tabs = s.tabs("S2"); m = s.flat_margins(p, tabs)
    mask = (s.idx[0] == 1) & (s.idx[2] == 2)
    lo_r, hi_r = subgroup_rate_bounds(s, tabs, m, m, mask)
    true = p.ravel()[mask & (s.idx[s.L] == 1)].sum() / p.ravel()[mask].sum()
    assert lo_r - 1e-7 <= true <= hi_r + 1e-7
    assert hi_r - lo_r > 0.05
