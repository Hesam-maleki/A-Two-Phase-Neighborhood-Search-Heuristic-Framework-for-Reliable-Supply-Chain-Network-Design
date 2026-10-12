"""
h1_ccg_ns.py: heuristic H1 for RO-LNDP0. Phase 1 is an early-stopped C&CG
run; Phase 2 is the shared filter-LP neighborhood search of
phase2_filter_ns.py.
"""

import math
import time
import datetime

from docplex.mp.model import Model

from ro_lndp0_ccg import (
    LNDPInstance,
    compute_big_M,
    solve_subproblem,
    build_master,
    add_ccg_cut,
    solve_mp_lp,
)
from instance_generator_paper_exact import (
    generate_instance_paper_exact,
    summarize_instance,
)
from phase2_filter_ns import run_phase2


# PHASE 1: Standard C&CG

def run_phase1_ccg(inst: LNDPInstance,
                   big_M: dict,
                   switch_gap_pct: float = 5.0,
                   phase1_time_limit: float = 1800.0,
                   eps: float = 1e-4,
                   use_lp_enhancement: bool = True,
                   verbose: bool = True):
    """
    Run standard C&CG until gap <= switch_gap_pct or phase1_time_limit.

    Returns
    -------
    y_best, UB, LB, scenarios_set, scenarios_list, history, status,
    mp, y_vars, phi_var
    """
    t_start = time.time()
    LB = -math.inf
    UB = math.inf
    y_best = None
    scenarios_set = set()
    scenarios_list = []
    history = []
    r = 0
    status = "SWITCHED"

    mp, y_vars, phi_var = build_master(inst)
    y_hat = {j: 1 for j in inst.V0}

    if verbose:
        print("\n" + "=" * 72)
        print("  PHASE 1 - Standard C&CG")
        print(f"  Switch condition : gap <= {switch_gap_pct:.1f}%  |  "
              f"Time budget : {phase1_time_limit:.0f}s")
        print("=" * 72)
        hdr = (f"{'Iter':>4}  {'LB':>14}  {'UB':>14}  {'Gap%':>8}  "
               f"{'SubP(s)':>8}  {'MP(s)':>8}")
        print(hdr)
        print("-" * len(hdr))

    while True:
        elapsed = time.time() - t_start

        if elapsed >= phase1_time_limit:
            status = "TIME_LIMIT"
            if verbose:
                print(f"  [Phase 1 time limit reached: {elapsed:.1f}s]")
            break

        iter_t0 = time.time()

        remaining = max(10.0, phase1_time_limit - elapsed)
        subp_t0 = time.time()
        z_hat, chi = solve_subproblem(inst, y_hat, big_M, remaining)
        subp_time = time.time() - subp_t0
        r += 1

        z_key = frozenset(j for j in inst.V0 if z_hat.get(j, 0) == 1)
        if z_key not in scenarios_set:
            scenarios_set.add(z_key)
            scenarios_list.append(dict(z_hat))

        add_ccg_cut(mp, inst, y_vars, phi_var, z_hat, r)

        fc_hat = sum(inst.f[j] * y_hat[j] for j in inst.V0)
        UB_cand = fc_hat + chi
        if UB_cand < UB:
            UB = UB_cand
            y_best = dict(y_hat)

        master_t0 = time.time()
        if use_lp_enhancement:
            lb_lp = solve_mp_lp(mp, y_vars)
            LB = max(LB, lb_lp)

        sol_mp = mp.solve(log_output=False)
        master_time = time.time() - master_t0

        if sol_mp:
            LB = max(LB, mp.objective_value)
            y_hat = {j: int(round(y_vars[j].solution_value)) for j in inst.V0}

        gap = (UB - LB) / abs(UB) * 100 if UB not in (0.0, math.inf) else math.inf
        elapsed = time.time() - t_start

        history.append(dict(
            iter          = r,
            LB            = LB,
            UB            = UB,
            gap           = gap,
            time_elapsed  = elapsed,
            subp_time     = subp_time,
            master_time   = master_time,
            iter_time     = time.time() - iter_t0,
            subp_status   = "optimal",
            master_status = "optimal" if sol_mp else "no_sol",
            z_hat         = dict(z_hat),
            y_hat         = dict(y_hat),
            phase         = 1,
        ))

        if verbose:
            print(f"{r:>4}  {LB:>14.2f}  {UB:>14.2f}  {gap:>7.4f}%  "
                  f"{subp_time:>8.2f}  {master_time:>8.2f}")

        if gap <= eps * 100:
            status = "OPTIMAL"
            if verbose:
                print(f"\n  Phase 1 converged exactly  (gap={gap:.6f}%)")
            break

        if gap <= switch_gap_pct:
            status = "SWITCHED"
            if verbose:
                print(f"\n  Gap {gap:.4f}% <= {switch_gap_pct}%  "
                      f"-> switching to Phase 2")
            break

    return (y_best, UB, LB,
            scenarios_set, scenarios_list,
            history, status,
            mp, y_vars, phi_var)


# H1: main entry point

def h1_ccg_ns(inst: LNDPInstance,
              switch_gap_pct: float = 10.0,
              phase1_time_limit: float = 10800.0,
              phase2_time_limit: float = 10800.0,
              eps: float = 1e-6,
              use_lp_enhancement: bool = True,
              pool_cap: int = 0,
              verbose: bool = True):
    """
    H1: early-stopped C&CG followed by the shared Phase 2.

    pool_cap caps the number of scenarios used in the Phase 2 filter
    LP per sweep (most recent kept); pool_cap = 0 means unlimited.

    Returns
    -------
    y_star, UB, LB, history, status
    """
    t_total = time.time()
    big_M = compute_big_M(inst)

    (y_p1, UB_p1, LB_p1,
     scenarios_set, scenarios_list,
     history_p1, p1_status,
     mp, y_vars, phi_var) = run_phase1_ccg(
        inst,
        big_M,
        switch_gap_pct     = switch_gap_pct,
        phase1_time_limit  = phase1_time_limit,
        eps                = eps,
        use_lp_enhancement = use_lp_enhancement,
        verbose            = verbose,
    )
    mp.end()

    _p1_iters = sum(1 for h in history_p1 if h.get("phase", 1) == 1)
    _p1_time = history_p1[-1]["time_elapsed"] if history_p1 else 0.0
    phase1 = dict(
        y       = dict(y_p1),
        UB      = UB_p1,
        LB      = LB_p1,
        status  = "OPTIMAL" if p1_status == "OPTIMAL" else "EARLY_STOP",
        iters   = _p1_iters,
        time    = _p1_time,
        history = history_p1,
    )

    if p1_status == "OPTIMAL":
        total_time = time.time() - t_total
        if verbose:
            _print_summary(y_p1, UB_p1, LB_p1, history_p1,
                           "OPTIMAL", total_time, inst, 0.0)
        return y_p1, UB_p1, LB_p1, history_p1, "OPTIMAL", phase1

    (y_p2, UB_p2,
     history_p2, p2_status) = run_phase2(
        inst,
        big_M,
        y_init         = y_p1,
        UB_init        = UB_p1,
        LB_fixed       = LB_p1,
        scenarios_set  = scenarios_set,
        scenarios_list = scenarios_list,
        time_limit     = phase2_time_limit,
        pool_cap       = pool_cap,
        iter_offset    = len(history_p1),
        verbose        = verbose,
    )

    history = history_p1 + history_p2
    UB_final = min(UB_p1, UB_p2)
    y_final = y_p2 if UB_p2 < UB_p1 else y_p1

    gap_final = (UB_final - LB_p1) / abs(UB_final) * 100 if UB_final > 0 else math.inf
    if gap_final <= eps * 100:
        final_status = "OPTIMAL"
    elif p2_status == "TIME_LIMIT":
        final_status = "TIME_LIMIT"
    else:
        final_status = "HEURISTIC"

    total_time = time.time() - t_total
    ub_improvement = UB_p1 - UB_p2

    if verbose:
        _print_summary(y_final, UB_final, LB_p1, history,
                       final_status, total_time, inst, ub_improvement)

    return y_final, UB_final, LB_p1, history, final_status, phase1


# Console summary

def _print_summary(y_star, UB, LB, history, status,
                   total_time, inst, ub_improvement):
    open_facs = [j for j in inst.V0 if y_star.get(j, 0) == 1]
    fc = sum(inst.f[j] * y_star.get(j, 0) for j in inst.V0)
    gap_pct = (UB - LB) / abs(UB) * 100 if UB > 0 else math.inf
    p1_iters = sum(1 for h in history if h.get("phase", 1) == 1)
    p2_moves = sum(1 for h in history if h.get("phase", 1) == 2)

    print("\n" + "=" * 65)
    print("  H1 - FINAL RESULTS")
    print("=" * 65)
    print(f"  Status                 : {status}")
    print(f"  Total time             : {total_time:.1f}s")
    print(f"  Phase 1 iterations     : {p1_iters}")
    print(f"  Phase 2 improvements   : {p2_moves}")
    print(f"  Upper bound (UB)       : {UB:,.2f}")
    print(f"  Lower bound (LB)       : {LB:,.2f}")
    print(f"  Optimality gap         : {gap_pct:.4f}%")
    print(f"  UB improvement (P2)    : {ub_improvement:,.2f}")
    print(f"  Fixed cost             : {fc:,.2f}")
    print(f"  Worst-case recourse    : {UB - fc:,.2f}")
    print(f"  Open facilities ({len(open_facs):>2})    : {open_facs}")
    print("=" * 65)


# Recourse helpers (used by Excel export)

def _compute_worst_case(inst, y, big_M, time_limit=300.0):
    """Return (z_hat, recourse_cost, fixed_cost) for the given solution y."""
    z_hat, chi = solve_subproblem(inst, y, big_M, time_limit)
    fc = sum(inst.f[j] * y.get(j, 0) for j in inst.V0)
    return z_hat, chi, fc


def _solve_recourse_lp(inst, y, z):
    """
    Solve the recourse LP for a fixed y and disruption z.
    Returns (recourse_cost, unmet_total, unmet_by_node_dict).
    """
    lp = Model(name="recourse_lp", log_output=False)
    x = {(i, j): lp.continuous_var(lb=0) for (i, j) in inst.arcs}
    u = {j: lp.continuous_var(lb=0) for j in inst.VD}

    for j in inst.VS:
        lp.add_constraint(lp.sum(x[(j, k)] for k in inst.Vplus[j]) <= inst.b[j])
    for j in inst.VT:
        lp.add_constraint(
            lp.sum(x[(i, j)] for i in inst.Vminus[j])
            == lp.sum(x[(j, k)] for k in inst.Vplus[j]))
    for j in inst.VD:
        lp.add_constraint(
            lp.sum(x[(i, j)] for i in inst.Vminus[j]) + u[j] == -inst.b[j])
    for j in inst.V0:
        cap = (1 - z.get(j, 0)) * inst.Q[j] * y.get(j, 0)
        lp.add_constraint(lp.sum(x[(j, k)] for k in inst.Vplus[j]) <= cap)

    lp.minimize(
        lp.sum(inst.c[(i, j)] * x[(i, j)] for (i, j) in inst.arcs)
        + lp.sum(inst.theta[j] * u[j] for j in inst.VD))

    sol = lp.solve(log_output=False)
    if not sol:
        lp.end()
        return math.inf, math.inf, {}

    recourse = lp.objective_value
    unmet_total = sum(u[j].solution_value for j in inst.VD)
    unmet_vals = {j: round(u[j].solution_value, 4)
                  for j in inst.VD if u[j].solution_value > 1e-6}
    lp.end()
    return recourse, unmet_total, unmet_vals


# Excel export

def export_h1_excel(inst, raw, gen_params,
                        y_star, UB, LB, history, status,
                        big_M, ub_improvement, total_time,
                        out_path):
    """
    Write a 5-sheet workbook:
      1. Summary
      2. Phase 1 C&CG history
      3. Phase 2 neighborhood search history
      4. Facility decisions
      5. Worst-case disruption detail
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        from openpyxl.chart import LineChart, Reference
    except ImportError:
        print("  [openpyxl not installed, skipping Excel export]")
        return

    HDR_FILL  = PatternFill("solid", fgColor="1F4E79")
    P1_FILL   = PatternFill("solid", fgColor="D6E4F0")
    P2_FILL   = PatternFill("solid", fgColor="FFF2CC")
    ALT_FILL  = PatternFill("solid", fgColor="F5F5F5")
    GOOD_FILL = PatternFill("solid", fgColor="C6EFCE")
    BAD_FILL  = PatternFill("solid", fgColor="FFCCCC")
    IMPR_FILL = PatternFill("solid", fgColor="FFE599")
    HDR_FONT  = Font(bold=True, color="FFFFFF", size=11)
    BOLD      = Font(bold=True, size=10)
    NORM      = Font(size=10)
    thin      = Side(style="thin", color="CCCCCC")
    BORDER    = Border(left=thin, right=thin, top=thin, bottom=thin)
    CENTER    = Alignment(horizontal="center", vertical="center", wrap_text=True)
    LEFT      = Alignment(horizontal="left",   vertical="center", wrap_text=True)
    NUM2      = '#,##0.00'
    NUM6      = '0.000000'

    def cell(ws, r, c, val, font=None, fill=None, align=None, num_fmt=None):
        cl = ws.cell(row=r, column=c, value=val)
        cl.border = BORDER
        if font:    cl.font          = font
        if fill:    cl.fill          = fill
        if align:   cl.alignment     = align
        if num_fmt: cl.number_format = num_fmt
        return cl

    def hrow(ws, r, labels, fills=None):
        for ci, lbl in enumerate(labels, 1):
            f = (fills[ci - 1] if fills and ci - 1 < len(fills) else HDR_FILL)
            cell(ws, r, ci, lbl, font=HDR_FONT, fill=f, align=CENTER)
        ws.row_dimensions[r].height = 22

    def auto_col(ws):
        for col in ws.columns:
            mx = max((len(str(cl.value or "")) for cl in col), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(mx + 4, 45)

    def section_hdr(ws, r, text, ncols):
        ws.merge_cells(start_row=r, start_column=1,
                       end_row=r, end_column=ncols)
        cl = ws.cell(row=r, column=1, value=text)
        cl.font      = Font(bold=True, size=10, color="1F4E79")
        cl.fill      = PatternFill("solid", fgColor="EBF3FB")
        cl.alignment = LEFT
        cl.border    = BORDER
        ws.row_dimensions[r].height = 18

    p1_hist = [h for h in history if h.get("phase", 1) == 1]
    p2_hist = [h for h in history if h.get("phase", 1) == 2]
    p1_iters = len(p1_hist)
    p2_moves = len(p2_hist)
    open_facs = sorted(j for j in inst.V0 if y_star.get(j, 0) == 1)
    fc = sum(inst.f[j] * y_star.get(j, 0) for j in inst.V0)
    gap_pct = (UB - LB) / abs(UB) * 100 if UB > 0 else math.inf

    z_worst, recourse, _ = _compute_worst_case(inst, y_star, big_M)
    _, unmet_total, unmet_vals = _solve_recourse_lp(inst, y_star, z_worst)
    disrupted_facs = sorted(j for j in inst.V0 if z_worst.get(j, 0) == 1)

    instance_label = (
        f"{int(gen_params['density']*100)}%-"
        f"{gen_params['nS']}-{gen_params['nT']}-{gen_params['nD']}  "
        f"seed={gen_params['seed']}  m={gen_params['m_budget']}"
    )

    wb = openpyxl.Workbook()

    # SHEET 1: Summary
    ws = wb.active
    ws.title = "Summary"
    ws.freeze_panes = "B3"

    ws.merge_cells("A1:C1")
    cl = ws.cell(row=1, column=1,
                 value=f"H1 (early-stopped C&CG + Phase 2)  |  {instance_label}")
    cl.font      = Font(bold=True, size=13, color="FFFFFF")
    cl.fill      = HDR_FILL
    cl.alignment = CENTER
    ws.row_dimensions[1].height = 30

    hrow(ws, 2, ["Category", "Metric", "Value"])
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 32
    ws.column_dimensions["C"].width = 22

    rows = [
        ("Instance",   "Nodes (total)",              len(inst.V)),
        ("Instance",   "Supply nodes |VS|",          len(inst.VS)),
        ("Instance",   "Transshipment nodes |VT|",   len(inst.VT)),
        ("Instance",   "Demand nodes |VD|",          len(inst.VD)),
        ("Instance",   "Facility nodes |V0|",        len(inst.V0)),
        ("Instance",   "Arcs |A|",                   len(inst.arcs)),
        ("Instance",   "Disruption budget m",        inst.m),
        ("Instance",   "Total supply",               round(sum(inst.b[j] for j in inst.VS), 2)),
        ("Instance",   "Total demand",               round(sum(-inst.b[j] for j in inst.VD), 2)),
        (None, None, None),
        ("Phase 1",    "C&CG iterations",            p1_iters),
        ("Phase 1",    "Phase 1 switch gap %",       round(gen_params.get("switch_gap_pct", 5.0), 2)),
        ("Phase 1",    "LB at switch",               round(LB, 2)),
        ("Phase 1",    "UB at switch",               round(p1_hist[-1]["UB"] if p1_hist else UB, 2)),
        ("Phase 1",    "Gap at switch %",            round(p1_hist[-1]["gap"] if p1_hist else gap_pct, 4)),
        (None, None, None),
        ("Phase 2",    "NS sweeps / improvements",   p2_moves),
        ("Phase 2",    "UB improvement by NS",       round(ub_improvement, 2)),
        ("Phase 2",    "UB improvement %",           round(ub_improvement / (UB + ub_improvement) * 100
                                                          if (UB + ub_improvement) > 0 else 0, 4)),
        (None, None, None),
        ("Solution",   "Status",                     status),
        ("Solution",   "Total solve time (s)",       round(total_time, 2)),
        ("Solution",   "Upper bound (UB)",           round(UB, 2)),
        ("Solution",   "Lower bound (LB)",           round(LB, 2)),
        ("Solution",   "Optimality gap %",           round(gap_pct, 6)),
        ("Solution",   "Fixed cost",                 round(fc, 2)),
        ("Solution",   "Worst-case recourse",        round(recourse, 2)),
        ("Solution",   "Recourse / UB %",            round(recourse / UB * 100 if UB > 0 else 0, 2)),
        ("Solution",   "Unmet demand (units)",       round(unmet_total, 2)),
        ("Solution",   "Open facilities (count)",    len(open_facs)),
        ("Solution",   "Open facilities (list)",     str(open_facs)),
        (None, None, None),
        ("Disruption", "Worst-case disrupted facs",  str(disrupted_facs)),
        ("Disruption", "Num disrupted",              len(disrupted_facs)),
    ]

    ri = 3
    for cat, metric, val in rows:
        if cat is None:
            ri += 1
            continue
        bg = P1_FILL if cat == "Phase 1" else (P2_FILL if cat == "Phase 2" else ALT_FILL if ri % 2 == 0 else None)
        cell(ws, ri, 1, cat,    font=BOLD, fill=bg, align=LEFT)
        cell(ws, ri, 2, metric, font=NORM, fill=bg, align=LEFT)
        num_fmt = NUM2 if isinstance(val, float) else None
        cell(ws, ri, 3, val,    font=NORM, fill=bg, align=CENTER, num_fmt=num_fmt)
        ws.row_dimensions[ri].height = 18
        ri += 1

    # SHEET 2: Phase 1 history
    ws2 = wb.create_sheet("Phase 1 - C&CG History")
    ws2.freeze_panes = "A2"

    cols2 = ["Iter", "LB", "UB", "Gap %",
             "SubP (s)", "MP (s)", "Iter (s)", "Elapsed (s)",
             "Disrupted facilities (z_hat)"]
    hrow(ws2, 1, cols2, fills=[P1_FILL] * len(cols2))

    for ri, h in enumerate(p1_hist, 2):
        z = h.get("z_hat", {})
        dis = sorted(j for j in inst.V0 if z.get(j, 0) == 1)
        bg = ALT_FILL if ri % 2 == 0 else None
        vals = [
            h.get("iter"),
            h.get("LB"),
            h.get("UB"),
            h.get("gap"),
            round(h.get("subp_time",    0), 3),
            round(h.get("master_time",  0), 3),
            round(h.get("iter_time",    0), 3),
            round(h.get("time_elapsed", 0), 3),
            str(dis),
        ]
        fmts = [None, NUM2, NUM2, NUM6, NUM2, NUM2, NUM2, NUM2, None]
        for ci, (v, fmt) in enumerate(zip(vals, fmts), 1):
            cell(ws2, ri, ci, v, fill=bg, align=CENTER, num_fmt=fmt)
        ws2.row_dimensions[ri].height = 16

    try:
        chart = LineChart()
        chart.title        = "Phase 1 - LB and UB convergence"
        chart.y_axis.title = "Cost"
        chart.x_axis.title = "Iteration"
        chart.style  = 10
        chart.width  = 20
        chart.height = 12

        n = len(p1_hist)
        lb_ref = Reference(ws2, min_col=2, min_row=1, max_row=n + 1)
        ub_ref = Reference(ws2, min_col=3, min_row=1, max_row=n + 1)
        chart.add_data(lb_ref, titles_from_data=True)
        chart.add_data(ub_ref, titles_from_data=True)
        chart.series[0].graphicalProperties.line.solidFill = "2E75B6"
        chart.series[1].graphicalProperties.line.solidFill = "C00000"
        ws2.add_chart(chart, f"A{n + 5}")
    except Exception:
        pass

    auto_col(ws2)

    # SHEET 3: Phase 2 history
    ws3 = wb.create_sheet("Phase 2 - NS History")
    ws3.freeze_panes = "A2"

    if p2_hist:
        cols3 = ["Move #", "UB after move", "LB (frozen)",
                 "Gap %", "SubP (s)", "Elapsed (s)", "Move description"]
        hrow(ws3, 1, cols3, fills=[P2_FILL] * len(cols3))

        for ri, h in enumerate(p2_hist, 2):
            bg = IMPR_FILL
            vals = [
                h.get("iter"),
                h.get("UB"),
                h.get("LB"),
                h.get("gap"),
                round(h.get("subp_time",    0), 3),
                round(h.get("time_elapsed", 0), 3),
                h.get("move_type", ""),
            ]
            fmts = [None, NUM2, NUM2, NUM6, NUM2, NUM2, None]
            for ci, (v, fmt) in enumerate(zip(vals, fmts), 1):
                cell(ws3, ri, ci, v, fill=bg, align=CENTER, num_fmt=fmt)
            ws3.row_dimensions[ri].height = 16
    else:
        ws3.cell(row=2, column=1,
                 value="Phase 2 did not find any improving moves.")

    auto_col(ws3)

    # SHEET 4: Facilities
    ws4 = wb.create_sheet("Facilities")
    ws4.freeze_panes = "A2"

    cols4 = ["Facility ID", "Type",
             "Fixed cost", "Capacity",
             "Open (final)", "Disrupted (worst-case)",
             "Open AND disrupted?"]
    hrow(ws4, 1, cols4)

    for ri, j in enumerate(sorted(inst.V0), 2):
        ftype   = "Supply" if j in inst.VS else "Transshipment"
        is_open = y_star.get(j, 0) == 1
        is_dis  = z_worst.get(j, 0) == 1
        both    = is_open and is_dis

        bg = ALT_FILL if ri % 2 == 0 else None
        vals = [j, ftype,
                round(inst.f[j], 2), round(inst.Q[j], 2),
                "OPEN" if is_open else "closed",
                "FAIL" if is_dis  else "ok",
                "YES - disrupted while open" if both else ("-" if not is_open else "safe")]
        fmts = [None, None, NUM2, NUM2, None, None, None]

        for ci, (v, fmt) in enumerate(zip(vals, fmts), 1):
            fill = bg
            if ci == 5:
                fill = GOOD_FILL if is_open else BAD_FILL
            elif ci == 6:
                fill = BAD_FILL  if is_dis  else GOOD_FILL
            elif ci == 7 and both:
                fill = IMPR_FILL
            cell(ws4, ri, ci, v, fill=fill, align=CENTER, num_fmt=fmt)
        ws4.row_dimensions[ri].height = 16

    auto_col(ws4)

    # SHEET 5: Disruption detail
    ws5 = wb.create_sheet("Disruption Detail")

    section_hdr(ws5, 1, "Worst-case disruption scenario for the final solution", 4)

    hrow(ws5, 2, ["Metric", "Value", "", ""], fills=[HDR_FILL, HDR_FILL, HDR_FILL, HDR_FILL])
    summary_rows = [
        ("Disrupted facilities", str(disrupted_facs)),
        ("Number disrupted",     len(disrupted_facs)),
        ("Recourse cost",        round(recourse, 2)),
        ("Total unmet demand",   round(unmet_total, 2)),
        ("Fixed cost",           round(fc, 2)),
        ("Total cost (UB)",      round(UB, 2)),
        ("Optimality gap %",     round(gap_pct, 4)),
    ]
    for ri, (metric, val) in enumerate(summary_rows, 3):
        bg = ALT_FILL if ri % 2 == 0 else None
        cell(ws5, ri, 1, metric, font=BOLD, fill=bg, align=LEFT)
        num_fmt = NUM2 if isinstance(val, float) else None
        cell(ws5, ri, 2, val, font=NORM, fill=bg, align=CENTER, num_fmt=num_fmt)
        ws5.row_dimensions[ri].height = 16

    start_row = len(summary_rows) + 5
    section_hdr(ws5, start_row, "Unmet demand by demand node", 4)
    hrow(ws5, start_row + 1,
         ["Demand node", "Required demand", "Unmet demand", "Unmet %"],
         fills=[HDR_FILL] * 4)

    for ri, d in enumerate(sorted(inst.VD), start_row + 2):
        required = round(-inst.b[d], 2)
        unmet    = round(unmet_vals.get(d, 0.0), 2)
        pct      = round(unmet / required * 100 if required > 0 else 0, 1)
        bg       = ALT_FILL if ri % 2 == 0 else None
        fill_u   = BAD_FILL if unmet > 1e-4 else GOOD_FILL
        vals = [d, required, unmet, pct]
        fmts = [None, NUM2, NUM2, '0.0"%"']
        for ci, (v, fmt) in enumerate(zip(vals, fmts), 1):
            fill = fill_u if ci in (3, 4) else bg
            cell(ws5, ri, ci, v, fill=fill, align=CENTER, num_fmt=fmt)
        ws5.row_dimensions[ri].height = 16

    auto_col(ws5)
    ws5.column_dimensions["A"].width = 16
    ws5.column_dimensions["B"].width = 18
    ws5.column_dimensions["C"].width = 16
    ws5.column_dimensions["D"].width = 12

    wb.save(out_path)
    print(f"\n  Excel report saved -> {out_path}")


# MAIN

def main():
    print("=" * 65)
    print("  RO-LNDP0  H1 (early-stopped C&CG + Phase 2)")
    print("=" * 65)

    GEN_PARAMS = dict(
        nS       = 10,
        nT       = 20,
        nD       = 30,
        density  = 0.20,
        seed     = 4,
        m_budget = 5,
    )

    SWITCH_GAP_PCT = 10.0
    TOTAL_TIME     = 10800.0
    EPS            = 1e-6
    POOL_CAP       = 0

    print(f"\nGenerating instance: {GEN_PARAMS}")
    raw = generate_instance_paper_exact(**GEN_PARAMS)
    summarize_instance(raw)
    inst = LNDPInstance.from_generator_dict(raw)

    print(f"\nNetwork: {len(inst.V)} nodes | {len(inst.arcs)} arcs | "
          f"{len(inst.V0)} facility nodes | m={inst.m}")

    t0 = time.time()
    y_star, UB, LB, history, status, _phase1 = h1_ccg_ns(
        inst,
        switch_gap_pct     = SWITCH_GAP_PCT,
        phase1_time_limit  = TOTAL_TIME,
        phase2_time_limit  = TOTAL_TIME,
        eps                = EPS,
        use_lp_enhancement = True,
        pool_cap           = POOL_CAP,
        verbose            = True,
    )
    total_time = time.time() - t0

    p1_ub = next(
        (h["UB"] for h in reversed(history) if h.get("phase", 1) == 1),
        UB
    )
    ub_improvement = p1_ub - UB

    big_M = compute_big_M(inst)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (f"{int(GEN_PARAMS['density']*100)}pct-"
             f"{GEN_PARAMS['nS']}-{GEN_PARAMS['nT']}-{GEN_PARAMS['nD']}")
    out_path = (f"H1_{label}_seed{GEN_PARAMS['seed']}"
                f"_m{GEN_PARAMS['m_budget']}_{ts}.xlsx")

    gen_params_full = dict(GEN_PARAMS, switch_gap_pct=SWITCH_GAP_PCT)

    export_h1_excel(
        inst           = inst,
        raw            = raw,
        gen_params     = gen_params_full,
        y_star         = y_star,
        UB             = UB,
        LB             = LB,
        history        = history,
        status         = status,
        big_M          = big_M,
        ub_improvement = ub_improvement,
        total_time     = total_time,
        out_path       = out_path,
    )


if __name__ == "__main__":
    main()
