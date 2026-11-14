"""Two guarantees an agency can compute BEFORE publishing a set of tables.

1. ``population_certificate``: worst-case AUC loss of the scorer implied by the tables (Theorem 4 / certificate).
2. ``subgroup_rate_bounds``:   exact lowest/highest outcome rate of a subgroup over ALL joints consistent with the
                               published tables (linear-fractional program via the Charnes-Cooper transform).
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack
from .metrics import cond_kl


def population_certificate(space, p, q):
    """Compare the scorer eta_q(x) = q(Y=1|x) with the best possible scorer eta(x) = p(Y=1|x) on population p.

    Returns a dict with
      regret         exact AUC(eta) - AUC(eta_q)
      certificate    the new bound  (1/(2 pi (1-pi))) E[ |eta(X)-eta(X')| * 1{ |eta(X)-eta(X')| <= d(X)+d(X') } ],  d = |eta-eta_q|
      bound_tv       (1/(pi(1-pi))) E d(X)                      (older, looser)
      bound_pinsker  sqrt(cond_kl/2) / (pi(1-pi))               (older, loosest)
    Proof of the certificate: AUC regret = (1/(2pi(1-pi))) E[|eta(X)-eta(X')| 1{eta_q misorders the pair}] and a pair can only
    be misordered if |eta(X)-eta(X')| <= d(X)+d(X')."""
    pi = p[..., 1].sum()
    px = p.sum(-1).ravel()
    kx = px > 0
    p1, p0, w = p[..., 1].ravel()[kx], p[..., 0].ravel()[kx], px[kx]
    eta = p1 / w
    qx = q.sum(-1).ravel()[kx]
    etaq = q[..., 1].ravel()[kx] / np.maximum(qx, 1e-300)
    d = np.abs(eta - etaq)

    def auc(s):
        gt = (s[:, None] > s[None, :]).astype(float) + 0.5 * (s[:, None] == s[None, :])
        return float((p1[:, None] * p0[None, :] * gt).sum() / (pi * (1 - pi)))

    D = np.abs(eta[:, None] - eta[None, :])
    W = w[:, None] * w[None, :]
    ck = cond_kl(p, q)
    return dict(
        regret=auc(eta) - auc(etaq),
        certificate=float((W * D * (D <= d[:, None] + d[None, :])).sum() / (2 * pi * (1 - pi))),
        bound_tv=float((w * d).sum() / (pi * (1 - pi))),
        bound_pinsker=float(np.sqrt(ck / 2) / (pi * (1 - pi))),
        cond_kl=ck, pi=float(pi))


def subgroup_rate_bounds(space, tabs, lo, hi, subgroup):
    """Lowest and highest P(Y=1 | subgroup) over all joint distributions whose margins (for ``tabs``) lie in [lo, hi].

    lo, hi   : stacked flattened margins in PROBABILITY units (use lo == hi for exact tables)
    subgroup : boolean mask over the flattened joint cells (e.g. X_i = a and X_j = b)
    Returns (rate_low, rate_high); raises if the program is infeasible."""
    A = space.A(tabs)
    ncell = A.shape[1]
    ms = subgroup.astype(float)
    my = (subgroup & (space.idx[space.L] == 1)).astype(float)
    exact = np.allclose(lo, hi)
    if exact:
        Aeq0 = hstack([A, csr_matrix(-lo[:, None])]).tocsr()
        Aub = bub = None
    else:
        Aeq0 = None
        Aub = vstack([hstack([-A, csr_matrix(lo[:, None])]), hstack([A, csr_matrix(-hi[:, None])])]).tocsr()
        bub = np.zeros(Aub.shape[0])
    row_tot = csr_matrix(np.r_[np.ones(ncell), -1.0][None, :])      # sum z = u
    row_S = csr_matrix(np.r_[ms, 0.0][None, :])                      # sum_{S} z = 1  (Charnes-Cooper normalisation)
    Aeq = vstack([Aeq0, row_tot, row_S]).tocsr() if Aeq0 is not None else vstack([row_tot, row_S]).tocsr()
    beq = np.r_[np.zeros(Aeq.shape[0] - 1), 1.0]
    out = []
    for sgn in (1, -1):
        c = np.r_[-sgn * my, 0.0]
        res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        out.append(-sgn * res.fun)
    return out[1], out[0]
