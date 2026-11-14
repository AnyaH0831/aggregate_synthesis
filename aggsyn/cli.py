"""Shared command-line options for the experiment scripts."""
import argparse
import os
import warnings
from .datasets import DATASETS


def make_parser(description, default_datasets=None, default_reps=10):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", default="data", help="folder holding the dataset files (default: data)")
    p.add_argument("--out-dir", default="results", help="where result CSVs are written (default: results)")
    p.add_argument("--datasets", nargs="+", default=default_datasets or DATASETS, choices=DATASETS)
    p.add_argument("--reps", type=int, default=default_reps, help="number of random splits / repeats")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test run (1-2 repeats, small solver limits)")
    return p


def setup(args):
    warnings.filterwarnings("ignore")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.quick:
        args.reps = min(args.reps, 2)
    return args
