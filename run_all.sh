#!/usr/bin/env bash
# Reproduce every result file in results/. Full run takes roughly 1-2 hours on one CPU core.
#   ./run_all.sh            full run
#   ./run_all.sh --quick    smoke test (minutes)
set -e
DATA=${DATA_DIR:-data}
OUT=${OUT_DIR:-results}
EXTRA="$@"
for e in e00_theory_checks e10_auc_certificate e09_greedy_submodularity e01_baselines_ladder e02_rounding_m2 \
         e03_rounding_all_datasets e04_table_selection_population e05_lp_identification_bounds \
         e06_table_selection_sample e07_augmentation e08_privacy_risk_frontier; do
  echo "=== $e ==="
  python experiments/$e.py --data-dir "$DATA" --out-dir "$OUT" $EXTRA
done
python analysis/summarize.py "$OUT"
