# A Two-Phase Neighborhood Search Heuristic Framework for Reliable Supply Chain Network Design under Facility Disruptions

Code to reproduce **Table 2** of the manuscript (Maleki, Godichaud, Ouazene,
Bozorgi-Amiri; submitted to *Supply Chain Analytics*).

## Requirements

- Python 3.10
- IBM ILOG CPLEX Optimization Studio 22.1, academic or commercial edition.
  The free CPLEX Community Edition (`pip install cplex`) is limited to 1000
  variables and 1000 constraints and cannot solve these models.
- `pip install -r requirements.txt` (docplex, openpyxl)

To check that docplex finds the CPLEX engine:

```
python -c "from docplex.mp.environment import Environment; print(Environment.get_default_env().cplex_version)"
```

docplex locates the CPLEX engine through the `CPLEX_STUDIO_DIR2211`
environment variable set by the CPLEX installer, so `import cplex` failing
does not mean the setup is broken.

## Files

| File | Role |
|---|---|
| `reproduce_paper_table2.py` | **Entry point.** Runs the four methods on the 13 instances and writes Table 2 |
| `compare_all_methods.py` | Runs the four methods on one instance and exports a workbook |
| `ro_lndp0_ccg.py` | RO-LNDP0 model components and exact C&CG |
| `h1_ccg_ns.py` | H1: early-stopped C&CG (Phase 1) + Phase 2 |
| `h2_mip_ns.py` | H2: no-disruption MIP and one bootstrap subproblem (Phase 1) + Phase 2 |
| `phase2_filter_ns.py` | Phase 2 (filter LP and neighborhood search), shared by H1 and H2 |
| `instance_generator_paper_exact.py` | Instance generator (Cheng et al., 2018) |
| `rebuild_table2_csv.py` | Rebuilds `table2_combined.csv` of a run folder from its saved workbooks |

`reproduce_paper_table2.py` is the only entry point needed to reproduce the
paper. The `main()` functions of the other modules run a single instance
(20%-10-20-30, seed 4) with the same settings and are provided for testing.

## Instances

All instances use uncertainty budget m = 5.

| Instance | `nS` | `nT` | `nD` | `density` | `seed` | `m` |
|---|---|---|---|---|---|---|
| 20%-10-20-30 | 10 | 20 | 30 | 0.20 | 4  | 5 |
| 30%-10-20-30 | 10 | 20 | 30 | 0.30 | 4  | 5 |
| 50%-10-20-30 | 10 | 20 | 30 | 0.50 | 5  | 5 |
| 50%-20-20-30 | 20 | 20 | 30 | 0.50 | 23 | 5 |
| 50%-30-20-30 | 30 | 20 | 30 | 0.50 | 4  | 5 |
| 50%-40-20-30 | 40 | 20 | 30 | 0.50 | 37 | 5 |
| 30%-10-30-30 | 10 | 30 | 30 | 0.30 | 4  | 5 |
| 30%-10-40-30 | 10 | 40 | 30 | 0.30 | 6  | 5 |
| 30%-10-50-30 | 10 | 50 | 30 | 0.30 | 4  | 5 |
| 20%-10-20-40 | 10 | 20 | 40 | 0.20 | 4  | 5 |
| 20%-10-20-50 | 10 | 20 | 50 | 0.20 | 4  | 5 |
| 20%-10-20-60 | 10 | 20 | 60 | 0.20 | 4  | 5 |
| 50%-20-30-50 | 20 | 30 | 50 | 0.50 | 16 | 5 |

## Run

```
python reproduce_paper_table2.py
```

Settings: early-stop tolerance tau = 10%, exact tolerance eps = 1e-6, time
limit 10,800 s per method and instance (the current iteration is allowed to
finish).

**Running time:** about 11 hours in total, of which about 97% is spent on the
four hard instances (50%-20-20-30, 50%-30-20-30, 50%-40-20-30,
50%-20-30-50); the other nine instances take about 20 minutes together.

**Software:** Python 3.10, IBM ILOG CPLEX 22.1.1, docplex 2.31.254. Running
times depend on the machine; upper bounds and gaps do not.

An interrupted run can be continued with
`python reproduce_paper_table2.py --resume run_<timestamp>`.

## Output

Each run creates a new folder `table2_results/run_<timestamp>/` containing:

- `table2_combined.csv`: one row per instance plus an Average row
- `COMPARE_<instance>_seed<seed>_m5_<timestamp>.xlsx`: detailed results per instance

To rebuild the CSV of an existing run from its workbooks:
`python rebuild_table2_csv.py run_<timestamp>`.

### Mapping of `table2_combined.csv` to Table 2

| Table 2 | Exact C&CG | Early-stopped C&CG | H1 | H2 |
|---|---|---|---|---|
| Iter | `exact_iter` | `early_iter` | `H1_iter` | `H2_iter` |
| Time | `exact_time` | `early_time` | `H1_time` | `H2_time` |
| Gap | `exact_gap` | `early_gap` | `H1_gap` | not reported |
| UB | `exact_UB` | `early_UB` | `H1_UB` | `H2_UB` |
| Delta UB (%) | | | `deltaUB_H1_pct` | `deltaUB_H2_pct` |

### Definitions

- **Iter.** Exact and early-stopped C&CG: number of C&CG iterations. H1 and
  H2: number of accepted improving moves in Phase 2, that is, the number of
  sweeps that end with an accepted move. The final sweep, which finds no
  improving move and ends the search, is not counted.
- **Time.** Wall-clock seconds. For H1 it includes Phase 1; for H2 it
  includes the no-disruption MIP and the bootstrap subproblem.
- **Gap.** (UB - LB) / UB x 100. For exact and early-stopped C&CG, LB is the
  C&CG master bound. For H1, LB is the lower bound frozen at the end of
  Phase 1. For H2 the available bound (the no-disruption MIP objective) is
  too loose to be meaningful, so no gap is reported.
- **Delta UB.** (UB_H - UB_exact) / UB_exact x 100 for H = H1, H2. Negative values
  mean the heuristic found a better solution than exact C&CG within its time
  limit.
- **Early-stopped C&CG** is not a separate run. Its values are taken from
  Phase 1 of the same H1 run (C&CG stopped once the gap is at most tau = 10%),
  and its time is the Phase 1 time of H1.
