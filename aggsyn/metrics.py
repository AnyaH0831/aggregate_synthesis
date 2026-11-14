"""Population-level (exact table) quantities used by the theory: entropies, KL decompositions, AUC of a scorer."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def entropy(p) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def kl(p, q) -> float:
    """KL(p || q) in nats."""
    m = p > 0
    return float((p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))).sum())


def cond_kl(p, q) -> float:
    """Conditional KL  E_{x~p} KL( p(y|x) || q(y|x) ), with y on the LAST axis.
    Chain rule:  KL(p||q) = KL(p_X || q_X) + cond_kl(p, q)."""
    px = p.sum(-1, keepdims=True)
    qx = q.sum(-1, keepdims=True)
    eta = p / np.maximum(px, 1e-300)
    etaq = q / np.maximum(qx, 1e-300)
    m = p > 0
    return float((p[m] * (np.log(eta[m]) - np.log(np.maximum(etaq[m], 1e-300)))).sum())


def mutual_information(space, p, i, j) -> float:
    pij = space.marg(p, (i, j))
    a = pij.sum(1, keepdims=True)
    b = pij.sum(0, keepdims=True)
    m = pij > 0
    return float((pij[m] * np.log(pij[m] / (a * b)[m])).sum())


def weighted_auc(p, score) -> float:
    """AUC on the population p (y on the last axis) of a scorer given per-X-cell scores (flattened over features)."""
    pw = p.reshape(-1, 2)
    k = pw.sum(1) > 0
    y = np.r_[np.ones(k.sum()), np.zeros(k.sum())]
    return float(roc_auc_score(y, np.r_[score[k], score[k]], sample_weight=np.r_[pw[k, 1], pw[k, 0]]))


def population_lr_auc(space, p, q) -> float:
    """AUC on p of a main-effects logistic regression trained on the joint q (cells used as weighted samples)."""
    L = space.L
    X = np.hstack([np.eye(space.dims[i])[space.idx[i]] for i in range(L)])
    y = space.idx[L]
    w = q.ravel()
    m = w > 0
    clf = LogisticRegression(max_iter=3000).fit(X[m], y[m], sample_weight=w[m] / w.sum() * 1000)
    s = clf.predict_proba(X)[:, 1]
    pw = p.ravel()
    k = pw > 0
    return float(roc_auc_score(y[k], s[k], sample_weight=pw[k]))
