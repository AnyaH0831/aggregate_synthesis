"""Quickstart: choose which tables to publish, publish them rounded, rebuild people from the published numbers, check quality.
Run from the repo root:  python examples/quickstart.py --dataset lanark --data-dir data"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from aggsyn import datasets, TableSpace, Selector, population_certificate

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mroz", choices=datasets.DATASETS); ap.add_argument("--data-dir", default="data")
ap.add_argument("--k", type=int, default=3, help="number of pair-tables to publish"); ap.add_argument("--round", type=int, default=5, help="round to nearest b, hide cells < b")
a = ap.parse_args()

d = datasets.load(a.dataset, a.data_dir); s = TableSpace(d); S = Selector(s)
tr, te = next(StratifiedShuffleSplit(1, test_size=0.3, random_state=0).split(d, d.y))
train, test = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True); N = len(train)
t = s.table(train); p = t / N

# --- DATA HOLDER: pick tables with the task-aware rule, check the pre-release certificate, publish rounded counts ---
tables = S.greedy(p, a.k, "cond")                       # which pair-tables to publish (index L = outcome)
tabs = S.ones + tables
cert = population_certificate(s, p, S.fit(p, tabs))
print("variables:", s.cols); print("tables to publish:", [tuple(s.cols[i] for i in e) for e in tables])
print(f"certificate (worst-case AUC loss of the tables' own scorer, vs the best scorer on these features): {cert['certificate']:.3f}")
pub = s.publish(t, tabs, a.round)                        # rounded to b, cells below b suppressed

# --- DATA USER: rebuild people from the published numbers only ---
lo, hi = s.intervals(pub, a.round)
q = s.m2(tabs, lo, hi, N)                                # fit the joint using the ranges (or s.ipf(tabs, s.naive_targets(pub), N))
syn = s.ilp(q, N, tabs, lo, hi, interval=True, tl=10)    # integer people that stay inside every published range
print(f"synthetic people: {len(syn)}, share of published ranges contradicted: {s.viol(syn, tabs, lo, hi):.3f}")
auc_syn, _ = s.tstr(syn, test); auc_real, _ = s.tstr(train, test)
print(f"test AUC  trained on synthetic: {auc_syn:.3f}   trained on real rows: {auc_real:.3f}")
