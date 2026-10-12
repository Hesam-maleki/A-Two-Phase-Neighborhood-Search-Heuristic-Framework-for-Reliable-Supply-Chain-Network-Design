"""
reproduce_paper_table2.py: reproduce Table 2 of the manuscript. This is the
entry point for reproducing the paper's results.

The script regenerates the 13 benchmark instances of Table 2 with the
instance generator of Cheng et al. (2018), using the (nS, nT, nD, density,
seed, m) values listed in INSTANCES, and runs four methods on each
instance: Exact C&CG, Early-stop C&CG (tau = 10%), H1 and H2. The exact
method uses a relative gap tolerance of 1e-6, and every method has a time
limit of 10800 seconds.

Output: each run writes to a new folder table2_results/run_<timestamp>/
next to this script, containing table2_combined.csv (one row per instance
plus an Average row) and one COMPARE_*.xlsx workbook per instance.

Resume: an interrupted run can be continued with --resume; instances
already recorded in that folder's table2_combined.csv are skipped.

Threads: no thread limit is set, so CPLEX uses its default automatic
thread configuration. Running times therefore depend on the machine.

Usage:
    python reproduce_paper_table2.py
    python reproduce_paper_table2.py --resume run_<timestamp>
"""
import os
import sys
import csv
import math
import time
import argparse
import datetime

from instance_generator_paper_exact import generate_instance_paper_exact
from ro_lndp0_ccg import LNDPInstance
from compare_all_methods import run_comparison, export_comparison_excel

# Instance parameters used for Table 2 of the manuscript
INSTANCES = [
    # label            nS  nT  nD  density  seed  m_budget
    ("20%-10-20-30",   10, 20, 30, 0.20,     4,   5),
    ("30%-10-20-30",   10, 20, 30, 0.30,     4,   5),
    ("50%-10-20-30",   10, 20, 30, 0.50,     5,   5),
    ("50%-20-20-30",   20, 20, 30, 0.50,    23,   5),
    ("50%-30-20-30",   30, 20, 30, 0.50,     4,   5),
    ("50%-40-20-30",   40, 20, 30, 0.50,    37,   5),
    ("30%-10-30-30",   10, 30, 30, 0.30,     4,   5),
    ("30%-10-40-30",   10, 40, 30, 0.30,     6,   5),
    ("30%-10-50-30",   10, 50, 30, 0.30,     4,   5),
    ("20%-10-20-40",   10, 20, 40, 0.20,     4,   5),
    ("20%-10-20-50",   10, 20, 50, 0.20,     4,   5),
    ("20%-10-20-60",   10, 20, 60, 0.20,     4,   5),
    ("50%-20-30-50",   20, 30, 50, 0.50,    16,   5),
]

STOP_GAP_PCT = 10.0
EPS_EXACT    = 1e-6
TIME_LIMIT   = 10800.0
POOL_CAP     = 0

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table2_results")
COMBINED_CSV_NAME = "table2_combined.csv"

FIELDS = [
    "instance",
    "exact_iter", "exact_time", "exact_gap", "exact_UB",
    "early_iter", "early_time", "early_gap", "early_UB",
    "H1_iter", "H1_time", "H1_gap", "H1_UB",
    "H2_iter", "H2_time", "H2_UB",
    "deltaUB_H1_pct", "deltaUB_H2_pct",
]


def _load_done_labels(combined_csv):
    if not os.path.exists(combined_csv):
        return {}
    rows = {}
    with open(combined_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["instance"] != "Average":
                rows[row["instance"]] = row
    return rows


def _write_combined_csv(rows_by_label, combined_csv):
    rows = [rows_by_label[label] for label, *_ in INSTANCES if label in rows_by_label]
    with open(combined_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        if rows:
            avg = {"instance": "Average"}
            numeric_fields = [f for f in FIELDS if f != "instance"]
            for f in numeric_fields:
                vals = [float(r[f]) for r in rows if r.get(f) not in (None, "")]
                avg[f] = round(sum(vals) / len(vals), 4) if vals else ""
            w.writerow(avg)


def _row_from_results(label, results):
    c1, c2, c3, c4 = (results["case1"], results["case2"],
                      results["case3"], results["case4"])
    ub_exact = c1["UB"]

    def gap(c):
        return round(max(c["gap"], 0.0), 4)

    return {
        "instance":   label,
        "exact_iter": c1["iters"],
        "exact_time": round(c1["time"], 2),
        "exact_gap":  gap(c1),
        "exact_UB":   round(c1["UB"], 2),
        "early_iter": c2["iters"],
        "early_time": round(c2["time"], 2),
        "early_gap":  gap(c2),
        "early_UB":   round(c2["UB"], 2),
        "H1_iter":    c3["p2_sweeps"],   # Iter for H1 and H2 counts Phase-2 sweeps only, as in Table 2
        "H1_time":    round(c3["time"], 2),
        "H1_gap":     gap(c3),
        "H1_UB":      round(c3["UB"], 2),
        "H2_iter":    c4["p2_sweeps"],
        "H2_time":    round(c4["time"], 2),
        "H2_UB":      round(c4["UB"], 2),
        "deltaUB_H1_pct": round((c3["UB"] - ub_exact) / ub_exact * 100, 4) if ub_exact else "",
        "deltaUB_H2_pct": round((c4["UB"] - ub_exact) / ub_exact * 100, 4) if ub_exact else "",
    }


def _parse_args():
    ap = argparse.ArgumentParser(description="Reproduce Table 2 of the manuscript.")
    ap.add_argument("--resume", metavar="RUN_FOLDER",
                    help="name of an existing run folder under table2_results/ "
                         "to continue (e.g. run_2026-10-12_09-15-30)")
    return ap.parse_args()


def _run_dir(args):
    if args.resume:
        run_dir = os.path.join(RESULTS_ROOT, os.path.basename(args.resume.rstrip("/\\")))
        if not os.path.isdir(run_dir):
            sys.exit(f"--resume: no such run folder: {run_dir}")
        return run_dir
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, f"run_{stamp}")
    os.makedirs(run_dir)
    return run_dir


def main():
    args = _parse_args()
    run_dir = _run_dir(args)
    combined_csv = os.path.join(run_dir, COMBINED_CSV_NAME)
    rows_by_label = _load_done_labels(combined_csv)

    print("=" * 72)
    print("  REPRODUCING TABLE 2  (13 instances x 4 methods)")
    print(f"  tau={STOP_GAP_PCT}%  eps_exact={EPS_EXACT}  time_limit={TIME_LIMIT}s")
    print(f"  output folder: {run_dir}")
    print("=" * 72)

    for label, nS, nT, nD, density, seed, m_budget in INSTANCES:
        if label in rows_by_label:
            print(f"\n[skip] {label} (already in {combined_csv})")
            continue

        gen_params = dict(nS=nS, nT=nT, nD=nD, density=density,
                           seed=seed, m_budget=m_budget)
        print(f"\n{'#' * 72}\n  INSTANCE {label}  seed={seed}\n{'#' * 72}")

        raw = generate_instance_paper_exact(**gen_params)
        inst = LNDPInstance.from_generator_dict(raw)

        t0 = time.time()
        results = run_comparison(
            inst, raw,
            gen_params       = gen_params,
            stop_gap_pct     = STOP_GAP_PCT,
            time_limit_exact = TIME_LIMIT,
            time_limit_H1    = TIME_LIMIT,
            time_limit_H2    = TIME_LIMIT,
            eps_exact        = EPS_EXACT,
            pool_cap_H1      = POOL_CAP,
            pool_cap_H2      = POOL_CAP,
            verbose          = True,
        )
        wall = time.time() - t0
        print(f"\n[{label}] done in {wall:.1f}s wall-clock")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace("%", "pct")
        xlsx_path = os.path.join(run_dir, f"COMPARE_{safe_label}_seed{seed}_m{m_budget}_{ts}.xlsx")
        export_comparison_excel(results, inst, raw, gen_params, xlsx_path)

        rows_by_label[label] = _row_from_results(label, results)
        _write_combined_csv(rows_by_label, combined_csv)
        print(f"[{label}] combined table updated: {combined_csv}")

    print("\nAll instances processed. See", combined_csv)


if __name__ == "__main__":
    main()
