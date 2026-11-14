"""Turn results/*.csv into the tables quoted in docs/RESEARCH_REPORT.md.  Usage:  python analysis/summarize.py [results_dir]
Writes results/SUMMARY.md and prints it. Missing files are skipped."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
out = []


def section(title):
    out.append(f"\n## {title}\n")


def table(df, **kw):
    out.append(df.round(kw.get("nd", 3)).to_markdown() + "\n")


def load(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (x > 0).mean()


r = load("e00_ladder.csv")
if r is not None:
    section("E00 theory checks: exact tables (AUC of a model trained on the implied joint q vs on the real joint p)")
    table(r.set_index(["data", "tables"]))
    nb = load("e00_naive_bayes.csv"); table(nb.set_index("data"), nd=15)
    table(load("e00_separation.csv").set_index(["m", "table"]))

r = load("e01_baselines_ladder.csv")
if r is not None:
    section("E01 classic generators on the S1/S2/S3 ladder (mean over repeats)")
    table(r.groupby(["data", "scen", "method"])[["fit_err", "tvd3", "auc", "p10"]].mean())

r = load("e02_rounding_m2.csv")
if r is not None:
    section("E02 standard fitting vs interval fitting (M2) as rounding gets coarser")
    g = r.groupby(["data", "b", "method"])[["tvd3", "viol", "suppressed_share"]].mean()
    table(g)
    rows = []
    for (d, b), x in r.groupby(["data", "b"]):
        a = x[x.method == "IPF-naive+ILP"].set_index("rep").tvd3; m = x[x.method == "M2+ILP"].set_index("rep").tvd3
        dd = (a - m).dropna()
        rows.append(dict(data=d, b=b, mean_gain=dd.mean(), m2_wins=f"{int((dd > 0).sum())}/{len(dd)}"))
    table(pd.DataFrame(rows).set_index(["data", "b"]), nd=4)

r = load("e03_rounding_all_datasets.csv")
if r is not None:
    section("E03 rounding variants on all datasets (all-pairs tables)")
    ref = r[r.b == 0].groupby(["data", "method"])[["tvd3", "auc", "uniq"]].mean()
    table(ref)
    table(r[r.b > 0].groupby(["data", "b", "method"])[["tvd3", "viol", "auc", "uniq"]].mean())
    section("E03 range violations averaged over datasets")
    table(r[r.b > 0].groupby(["b", "method"])[["tvd3", "viol", "auc", "uniq"]].mean())
    rows = []
    for (d, b), x in r[r.b > 0].groupby(["data", "b"]):
        base = x[x.method == "IPF-naive+midILP"].set_index("rep").tvd3
        for m in ["M2+midILP", "M2+intervalILP", "IPF-naive+intervalILP"]:
            dd = (base - x[x.method == m].set_index("rep").tvd3).dropna()
            rows.append(dict(data=d, b=b, method=m, gain=dd.mean(), wins=f"{int((dd > 0).sum())}/{len(dd)}"))
    table(pd.DataFrame(rows).set_index(["data", "b", "method"]), nd=4)

r = load("e04_table_selection_population.csv")
if r is not None:
    section("E04 population-level table selection: AUC loss vs real (mean over datasets)")
    x = r[r.strategy != "_randdraw"].assign(auc_loss=lambda z: z.base_auc - z.auc)
    table(x.pivot_table(index="k", columns="strategy", values="auc_loss"))
    section("E04 conditional KL (task error) by strategy (mean over datasets)")
    table(x.pivot_table(index="k", columns="strategy", values="ckl"))
    a = r.assign(auc_loss=lambda z: z.base_auc - z.auc)
    rows = [dict(data=n, joint_KL_vs_auc_loss=spearmanr(g.kl, g.auc_loss)[0], cond_KL_vs_auc_loss=spearmanr(g.ckl, g.auc_loss)[0], n=len(g)) for n, g in a.groupby("data")]
    section("E04 Spearman correlation with AUC loss across all evaluated table sets")
    table(pd.DataFrame(rows).set_index("data"), nd=2)

r = load("e05_lp_identification_bounds.csv")
if r is not None:
    section("E05 LP identification bounds on subgroup outcome rates")
    r = r.assign(err=(r.maxent - r.true).abs(), certified=(r.lo > r.base) | (r.hi < r.base))
    out.append(f"Truth inside the bounds in every case: **{bool(r.inside.all())}** ({len(r)} cases)\n")
    table(r.groupby("scen").agg(width=("width", "mean"), maxent_error=("err", "mean"), certified=("certified", "mean")))
    table(r.groupby(["data", "scen"]).agg(width=("width", "mean"), maxent_error=("err", "mean"), certified=("certified", "mean")))

r = load("e06_table_selection_sample.csv")
if r is not None:
    section("E06 sample-level selection (n~500, tables rounded to 5, small cells hidden): real-train reference AUC")
    ref = r[r.strategy == "real_train(TRTR)"].groupby("data")[["auc_lr", "auc_xgb"]].mean()
    table(ref)
    x = r[r.strategy != "real_train(TRTR)"].copy()
    x["lr_loss"] = x.data.map(ref.auc_lr) - x.auc_lr
    x["xgb_loss"] = x.data.map(ref.auc_xgb) - x.auc_xgb
    x["direct_loss"] = x.data.map(ref.auc_lr) - x.auc_direct
    section("E06 average AUC loss vs real data (lower is better)")
    table(x.groupby(["strategy", "k"])[["lr_loss", "xgb_loss", "direct_loss"]].mean())
    section("E06 k=3, logistic AUC loss by dataset")
    table(x[x.k == 3].pivot_table(index="strategy", columns="data", values="lr_loss"))
    section("E06 paired comparison at k=3 (task-aware greedy vs others; positive = task-aware better), 95% bootstrap CI")
    rows = []
    for b in ["KL-greedy", "MI-tree (MST/Chow-Liu)", "Random", "Label-first"]:
        for col in ["lr_loss", "xgb_loss"]:
            A = x[(x.strategy == "Task-aware greedy") & (x.k == 3)].set_index(["data", "rep"])[col]
            B = x[(x.strategy == b) & (x.k == 3)].set_index(["data", "rep"])[col]
            m, lo, hi, w = boot_ci((B - A).dropna().values)
            rows.append(dict(versus=b, metric=col, gain=m, ci_low=lo, ci_high=hi, task_aware_better=w, runs=len((B - A).dropna())))
    table(pd.DataFrame(rows).set_index(["versus", "metric"]))

r = load("e07_augmentation.csv")
if r is not None:
    section("E07 augmentation: real only vs synthetic only vs real + synthetic (mean AUC)")
    cols = ["real_lr", "syn_lr", "comb_lr_0.5", "comb_lr_1", "comb_lr_2", "comb_lr_4", "comb_lr_8", "real_xgb", "syn_xgb", "comb_xgb_1", "comb_xgb_4"]
    table(r.groupby("n")[[c for c in cols if c in r]].mean())
    rows = []
    for n, x in r.groupby("n"):
        m, lo, hi, w = boot_ci((x.comb_lr_2 - x.real_lr).values)
        mx, lox, hix, wx = boot_ci((x.comb_xgb_4 - x.real_xgb).values)
        rows.append(dict(n=n, lr_gain=m, lr_ci_low=lo, lr_ci_high=hi, lr_better=w, xgb_gain=mx, xgb_ci_low=lox, xgb_ci_high=hix, xgb_better=wx))
    table(pd.DataFrame(rows).set_index("n"))
    section("E07 by dataset (logistic regression)")
    table(r.groupby(["data", "n"])[["real_lr", "comb_lr_2", "syn_lr", "full_real_lr"]].mean())

r = load("e08_privacy_risk_frontier.csv")
if r is not None:
    section("E08 utility vs small-cell budget R (AUC loss, mean over datasets x repeats)")
    table(r.pivot_table(index="strategy", columns="R", values="auc_loss"))
    section("E08 number of tables chosen")
    table(r.pivot_table(index="strategy", columns="R", values="k"), nd=1)

r = load("e09_greedy_submodularity.csv")
if r is not None:
    section("E09 greedy vs optimum, submodularity, monotonicity, and the a-priori bound for outcome x feature tables")
    table(r.set_index("data"))

r = load("e10_auc_certificate.csv")
if r is not None:
    section("E10 AUC-loss certificate vs exact loss")
    out.append(f"Certificate valid in **{int(r.valid.sum())} of {len(r)}** cases\n")
    table(r.groupby("tables")[["regret", "certificate", "bound_tv", "bound_pinsker"]].mean())
    table(r.set_index(["data", "tables"])[["regret", "certificate", "bound_tv", "bound_pinsker"]])

text = "# Summary of results (generated by analysis/summarize.py)\n" + "\n".join(out)
(R / "SUMMARY.md").write_text(text)
print(text)
