#!/usr/bin/env python3
"""
Aggregate success across MetaWorld tasks and seeds.

Assumes runs are stored under:
  exp/<exp_name>/metaworld_<task>/.../train.csv
where train.csv has a column `true_episode_success`.

Outputs per-task mean/std over seeds and overall mean of means/stds.
"""
import argparse
import glob
import os
import pandas as pd
import csv


DEFAULT_TASKS = [
    "hammer-v3-goal-observable",
    "push-wall-v3-goal-observable",
    "faucet-close-v3-goal-observable",
    "push-back-v3-goal-observable",
    "stick-pull-v3-goal-observable",
    "handle-press-side-v3-goal-observable",
    "push-v3-goal-observable",
    "shelf-place-v3-goal-observable",
    "window-close-v3-goal-observable",
    "peg-unplug-side-v3-goal-observable",
]


def load_last_success(csv_path: str, column: str) -> float:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise KeyError(f"{column} not in {csv_path}")
    series = df[column].dropna()
    if series.empty:
        raise ValueError(f"No values for {column} in {csv_path}")
    return float(series.iloc[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default="exp/gt_task_reward", help="Root exp dir")
    ap.add_argument("--column", default="true_episode_success", help="Metric column to aggregate")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Task names without 'metaworld_' prefix")
    ap.add_argument("--pattern", default="**/train.csv", help="Glob pattern under each task dir")
    ap.add_argument("--output-csv", default="summary.csv", help="Path to save CSV summary (default: summary.csv)")
    args = ap.parse_args()

    results = []
    for task in args.tasks:
        task_dir = os.path.join(args.exp_root, f"metaworld_{task}")
        csv_paths = glob.glob(os.path.join(task_dir, args.pattern), recursive=True)
        if not csv_paths:
            print(f"[WARN] No train.csv found for {task}")
            continue
        vals = []
        for p in csv_paths:
            try:
                vals.append(load_last_success(p, args.column))
            except Exception as e:
                print(f"[WARN] Skip {p}: {e}")
        if not vals:
            print(f"[WARN] No valid values for {task}")
            continue
        mean = sum(vals) / len(vals)
        std = float(pd.Series(vals).std(ddof=1)) if len(vals) > 1 else 0.0
        results.append((task, len(vals), mean, std))
        print(f"{task}: n={len(vals)}, mean={mean:.4f}, std={std:.4f}")

    if results:
        mean_of_means = sum(r[2] for r in results) / len(results)
        mean_of_stds = sum(r[3] for r in results) / len(results)
        print(f"\nAcross tasks: mean of means = {mean_of_means:.4f}, mean of stds = {mean_of_stds:.4f}")
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "n", "mean", "std"])
            for r in results:
                writer.writerow(r)
            writer.writerow([])
            writer.writerow(["mean_of_means", "", mean_of_means, ""])
            writer.writerow(["mean_of_stds", "", mean_of_stds, ""])
        print(f"Saved summary to {args.output_csv}")
    else:
        print("No results aggregated.")


if __name__ == "__main__":
    main()
