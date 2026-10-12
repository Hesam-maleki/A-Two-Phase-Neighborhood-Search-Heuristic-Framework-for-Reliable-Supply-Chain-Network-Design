"""
h2_mip_ns.py: heuristic H2 for RO-LNDP0. Phase 1 solves the no-disruption
MIP (G-LNDP) followed by one bootstrap subproblem; Phase 2 is the shared
filter-LP neighborhood search of phase2_filter_ns.py.
"""

import math
import time
import datetime

from docplex.mp.model import Model

from ro_lndp0_ccg import (
    LNDPInstance,
    compute_big_M,
    solve_subproblem,
)
from instance_generator_paper_exact import (
    generate_instance_paper_exact,
    summarize_instance,
)
from phase2_filter_ns import run_phase2


# PHASE 1: No-Disruption MIP  (binary y, z = 0)

def solve_nodisruption_mip(inst: LNDPInstance, verbose: bool = True):
    """
    Solve the Generic LNDP from Cheng (2018) Appendix A: binary y_j,
    no disruption (z = 0), so capacity is always full Q_j * y_j.

    Returns (y_mip, mip_obj, status, solve_time).
    """
    if verbose:
        print("\n" + "=" * 65)
        print("  PHASE 1 - No-Disruption MIP  (binary y, z=0)")
        print(f"  |V0|={len(inst.V0)}  m={inst.m}")
        print("=" * 65)

    mip = Model(name="nodisrupt_mip", log_output=False)

    y = {j: mip.binary_var(name=f"y_{j}") for j in inst.V0}
    x = {(i, j): mip.continuous_var(lb=0, name=f"x_{i}_{j}")
         for (i, j) in inst.arcs}
    u = {j: mip.continuous_var(lb=0, name=f"u_{j}") for j in inst.VD}

    for j in inst.VS:
        mip.add_constraint(mip.sum(x[(j, k)] for k in inst.Vplus[j]) <= inst.b[j])
    for j in inst.VT:
        mip.add_constraint(mip.sum(x[(i, j)] for i in inst.Vminus[j]) ==
                           mip.sum(x[(j, k)] for k in inst.Vplus[j]))
    for j in inst.VD:
        mip.add_constraint(mip.sum(x[(i, j)] for i in inst.Vminus[j])
                           + u[j] == -inst.b[j])
    for j in inst.V0:
        mip.add_constraint(mip.sum(x[(j, k)] for k in inst.Vplus[j])
                           <= inst.Q[j] * y[j])

    mip.minimize(
        mip.sum(inst.f[j]     * y[j]     for j in inst.V0) +
        mip.sum(inst.c[(i, j)] * x[(i, j)] for (i, j) in inst.arcs) +
        mip.sum(inst.theta[j] * u[j]     for j in inst.VD))

    t0 = time.time()
    sol = mip.solve(log_output=False)
    tmip = time.time() - t0

    if not sol:
        mip.end()
        if verbose:
            print("  MIP: INFEASIBLE or ERROR")
        return {j: 0 for j in inst.V0}, math.inf, "infeasible", tmip

    y_mip = {j: int(round(y[j].solution_value)) for j in inst.V0}
    mip_obj = mip.objective_value
    status = "optimal" if mip.solve_details.status == "optimal" else "feasible"
    mip.end()

    if verbose:
        open_list = sorted(j for j in inst.V0 if y_mip[j] == 1)
        print(f"  MIP solved in {tmip:.3f}s  |  obj = {mip_obj:,.2f}  "
              f"|  status = {status}")
        print(f"  Open facilities ({len(open_list)}): {open_list}")

    return y_mip, mip_obj, status, tmip


# H2: main entry point

def h2_mip_ns(inst: LNDPInstance,
              phase2_time_limit: float = 10800.0,
              pool_cap: int = 0,
              verbose: bool = True):
    """
    H2: no-disruption MIP start followed by the shared Phase 2.

    pool_cap caps the number of scenarios used in the Phase 2 filter
    LP per sweep (most recent kept); pool_cap = 0 means unlimited.

    Returns
    -------
    y_star, UB, LB, history_p1, history_p2, status
    """
    t_total = time.time()
    big_M = compute_big_M(inst)

    # Phase 1
    y_mip, p1_obj, p1_status, p1_time = solve_nodisruption_mip(
        inst, verbose=verbose)
    if p1_status == "infeasible":
        y_mip = {j: 1 for j in inst.V0}; p1_obj = 0.0

    n_open = sum(y_mip.values())
    minop = inst.m + 2
    if n_open < minop:
        closed = sorted(
            [(j, inst.f[j]) for j in inst.V0 if y_mip[j] == 0],
            key=lambda kv: kv[1])
        for j, _ in closed[:minop - n_open]:
            y_mip[j] = 1
        if verbose:
            print(f"  [MIP] Forced open to minimum {minop}")

    y_start = y_mip

    t_sub = time.time()
    z_init, chi = solve_subproblem(inst, y_start, big_M)
    subp_time_p1 = time.time() - t_sub
    fc_start = sum(inst.f[j] * y_start[j] for j in inst.V0)
    UB_start = fc_start + chi
    LB = p1_obj

    history_p1 = dict(
        method    = "mip_nodisrupt",
        mip_obj   = p1_obj,
        mip_time  = p1_time,
        y_start   = dict(y_start),
        n_open    = sum(y_start.values()),
        UB_start  = UB_start,
        fc_start  = fc_start,
        chi_start = chi,
        subp_time = subp_time_p1,
        z_init    = dict(z_init),
    )

    if verbose:
        print(f"\n  UB after MIP start : {UB_start:,.2f}")
        print(f"  MIP obj={p1_obj:,.2f}  fixed={fc_start:,.2f}  "
              f"recourse={chi:,.2f}  subp={subp_time_p1:.3f}s")

    # Seed scenario pool
    scenarios_set = set()
    scenarios_list = []
    z_key = frozenset(j for j in inst.V0 if z_init.get(j, 0) == 1)
    scenarios_set.add(z_key)
    scenarios_list.append(dict(z_init))

    # Phase 2
    y_star, UB_final, history_p2, status = run_phase2(
        inst, big_M,
        y_init         = y_start,
        UB_init        = UB_start,
        LB_fixed       = LB,
        scenarios_set  = scenarios_set,
        scenarios_list = scenarios_list,
        time_limit     = phase2_time_limit,
        pool_cap       = pool_cap,
        verbose        = verbose,
    )

    total_elapsed = time.time() - t_total

    if verbose:
        open_facs = sorted(j for j in inst.V0 if y_star[j] == 1)
        fc = sum(inst.f[j] * y_star[j] for j in inst.V0)
        gap_pct = ((UB_final - LB) / abs(UB_final) * 100
                   if UB_final > 0 else math.inf)
        print("\n" + "=" * 65)
        print("  H2 - FINAL RESULTS")
        print("=" * 65)
        print(f"  Status              : {status}")
        print(f"  Total time          : {total_elapsed:.1f}s")
        print(f"  UB at start (P1)    : {UB_start:,.2f}")
        print(f"  UB final (after NS) : {UB_final:,.2f}")
        print(f"  NS improvement      : {UB_start - UB_final:,.2f}")
        print(f"  P1 lower bound      : {LB:,.2f}")
        print(f"  Gap (UB-LB)/UB      : {gap_pct:.4f}%")
        print(f"  Fixed cost          : {fc:,.2f}")
        print(f"  Recourse cost       : {UB_final - fc:,.2f}")
        print(f"  NS improvements     : {len(history_p2)}")
        print(f"  Final pool size     : {len(scenarios_list)}")
        print(f"  Open ({len(open_facs):>2})           : {open_facs}")
        print("=" * 65)

    return y_star, UB_final, LB, history_p1, history_p2, status


# Flow solver  (used by Excel export)

def _solve_flows(inst: LNDPInstance, y: dict, z_hat: dict):
    """
    Solve min-cost flow LP for facility decisions y and disruption z_hat.
    Returns (flows, unmet).
    """
    lp = Model(name="flows", log_output=False)
    x = {(i, j): lp.continuous_var(lb=0) for (i, j) in inst.arcs}
    u = {j: lp.continuous_var(lb=0) for j in inst.VD}

    for j in inst.VS:
        lp.add_constraint(lp.sum(x[(j, k)] for k in inst.Vplus[j]) <= inst.b[j])
    for j in inst.VT:
        lp.add_constraint(lp.sum(x[(i, j)] for i in inst.Vminus[j]) ==
                          lp.sum(x[(j, k)] for k in inst.Vplus[j]))
    for j in inst.VD:
        lp.add_constraint(lp.sum(x[(i, j)] for i in inst.Vminus[j])
                          + u[j] == -inst.b[j])
    for j in inst.V0:
        out = lp.sum(x[(j, k)] for k in inst.Vplus[j])
        if y.get(j, 0) == 0 or z_hat.get(j, 0) == 1:
            lp.add_constraint(out == 0)
        else:
            lp.add_constraint(out <= inst.Q[j])

    lp.minimize(lp.sum(inst.c[(i, j)] * x[(i, j)] for (i, j) in inst.arcs)
                + lp.sum(inst.theta[j] * u[j] for j in inst.VD))
    sol = lp.solve(log_output=False)

    flows = {(i, j): max(0.0, x[(i, j)].solution_value)
             for (i, j) in inst.arcs} if sol else {(i, j): 0.0 for (i, j) in inst.arcs}
    unmet = {j: max(0.0, u[j].solution_value)
             for j in inst.VD} if sol else {j: -inst.b[j] for j in inst.VD}
    lp.end()
    return flows, unmet


# Excel export

def export_h2_excel(inst: LNDPInstance,
                 raw: dict,
                 gen_params: dict,
                 history_p1: dict,
                 history_p2: list,
                 y_star: dict,
                 UB: float,
                 LB: float,
                 status: str,
                 total_time: float,
                 out_path: str):
    """
    Export results to a 5-sheet workbook:
      1. Summary
      2. Phase 1 facility decisions
      3. Phase 2 NS log
      4. Facility decisions + worst-case disruption
      5. Flow details under worst-case disruption
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)

    C_DARK   = "1F3864"
    C_MID    = "2E75B6"
    C_LIGHT  = "D6E4F0"
    C_GREEN  = "E2EFDA"
    C_RED    = "FFE0E0"
    C_GRAY   = "F2F2F2"
    C_WHITE  = "FFFFFF"
    C_STAR   = "C6EFCE"

    def _f(bold=False, size=10, color="000000"):
        return Font(name="Arial", bold=bold, size=size, color=color)

    def _fill(hex_c):
        return PatternFill("solid", fgColor=hex_c)

    def _border():
        s = Side(style="thin", color="BFBFBF")
        return Border(left=s, right=s, top=s, bottom=s)

    def _w(ws, col, w):
        ws.column_dimensions[get_column_letter(col)].width = w

    def cell(ws, r, c, val, bg=C_WHITE, bold=False, color="000000",
             fmt=None, align="center"):
        ce = ws.cell(row=r, column=c, value=val)
        ce.font = _f(bold=bold, color=color)
        ce.fill = _fill(bg)
        ce.alignment = Alignment(horizontal=align, vertical="center",
                                 wrap_text=True)
        ce.border = _border()
        if fmt:
            ce.number_format = fmt
        return ce

    def title(ws, row, text, n_cols):
        ws.row_dimensions[row].height = 24
        c = ws.cell(row=row, column=1, value=text)
        c.font = _f(bold=True, size=13, color=C_WHITE)
        c.fill = _fill(C_DARK)
        c.alignment = Alignment(horizontal="center", vertical="center")
        ws.merge_cells(start_row=row, start_column=1,
                       end_row=row, end_column=n_cols)

    def section(ws, row, text, n_cols, bg=C_MID):
        ws.row_dimensions[row].height = 18
        c = ws.cell(row=row, column=1, value=text)
        c.font = _f(bold=True, size=11, color=C_WHITE)
        c.fill = _fill(bg)
        c.alignment = Alignment(horizontal="left", vertical="center")
        ws.merge_cells(start_row=row, start_column=1,
                       end_row=row, end_column=n_cols)

    def hdr_row(ws, row, labels, bg=C_LIGHT):
        for c, lbl in enumerate(labels, 1):
            cell(ws, row, c, lbl, bg=bg, bold=True)

    # SHEET 1: Summary
    ws1 = wb.create_sheet("Summary")
    label = (f"{int(gen_params['density']*100)}%-"
             f"{gen_params['nS']}-{gen_params['nT']}-{gen_params['nD']}")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    title(ws1, 1, f"H2 (MIP + Phase 2)  |  {label}  |  {ts}", 4)

    section(ws1, 2, "INSTANCE", 4)
    inst_rows = [
        ("Supply nodes |VS|",   len(inst.VS)),
        ("Transshipment |VT|",  len(inst.VT)),
        ("Demand nodes |VD|",   len(inst.VD)),
        ("Facility nodes |V0|", len(inst.V0)),
        ("Arcs |A|",            len(inst.arcs)),
        ("Disruption budget m", inst.m),
        ("Seed",                gen_params["seed"]),
        ("Density",             gen_params["density"]),
    ]
    for ri, (lbl, val) in enumerate(inst_rows, 3):
        bg = C_GRAY if ri % 2 == 0 else C_WHITE
        cell(ws1, ri, 1, lbl, bg=bg, bold=True, align="left")
        cell(ws1, ri, 2, val, bg=bg)

    r = 3 + len(inst_rows) + 1
    section(ws1, r, "ALGORITHM PARAMETERS", 4)
    r += 1
    param_rows = [
        ("Phase 1 method", "mip_nodisrupt"),
        ("Phase 2 time limit", f"{gen_params.get('phase2_time_limit', '-')}s"),
        ("Pool cap",       (gen_params.get("pool_cap", 0)
                            if gen_params.get("pool_cap", 0) > 0
                            else "unlimited")),
    ]
    for lbl, val in param_rows:
        bg = C_GRAY if r % 2 == 0 else C_WHITE
        cell(ws1, r, 1, lbl, bg=bg, bold=True, align="left")
        cell(ws1, r, 2, val, bg=bg)
        r += 1

    r += 1
    section(ws1, r, "PHASE 1 - NO-DISRUPTION MIP  (binary y, z=0)", 4)
    r += 1
    p1_rows = [
        ("MIP objective (no-disruption LB)", round(history_p1["mip_obj"], 2)),
        ("MIP solve time (s)",               round(history_p1["mip_time"], 4)),
        ("Open facilities (MIP exact)",      history_p1["n_open"]),
        ("UB after MIP start",               round(history_p1["UB_start"], 2)),
        ("Fixed cost (MIP)",                 round(history_p1["fc_start"], 2)),
        ("Recourse cost (MIP start)",        round(history_p1["chi_start"], 2)),
        ("Subproblem time (s)",              round(history_p1["subp_time"], 4)),
    ]
    for lbl, val in p1_rows:
        bg = C_GRAY if r % 2 == 0 else C_WHITE
        cell(ws1, r, 1, lbl, bg=bg, bold=True, align="left")
        cell(ws1, r, 2, val, bg=bg,
             fmt="#,##0.00" if isinstance(val, float) else None)
        r += 1

    r += 1
    section(ws1, r, "PHASE 2 - NEIGHBORHOOD SEARCH RESULTS", 4)
    r += 1
    fc_final = sum(inst.f[j] * y_star.get(j, 0) for j in inst.V0)
    gap_pct = ((UB - LB) / abs(UB) * 100 if UB > 0 else math.inf)
    open_facs = sorted(j for j in inst.V0 if y_star.get(j, 0) == 1)

    p2_rows = [
        ("Status",                status),
        ("Total time (s)",        round(total_time, 1)),
        ("NS improvements",       len(history_p2)),
        ("UB at start",           round(history_p1["UB_start"], 2)),
        ("UB final",              round(UB, 2)),
        ("NS improvement",        round(history_p1["UB_start"] - UB, 2)),
        ("G-LNDP lower bound",    round(LB, 2)),
        ("Gap (UB-LB)/UB %",      round(gap_pct, 4)),
        ("Fixed cost",            round(fc_final, 2)),
        ("Recourse cost",         round(UB - fc_final, 2)),
        ("Open facilities (#)",   len(open_facs)),
        ("Open facility list",    str(open_facs)),
    ]
    for lbl, val in p2_rows:
        bg = C_STAR if lbl in ("UB final", "NS improvement") else (
             C_GRAY if r % 2 == 0 else C_WHITE)
        cell(ws1, r, 1, lbl, bg=bg, bold=True, align="left")
        cell(ws1, r, 2, val, bg=bg,
             fmt="#,##0.00" if isinstance(val, float) else None)
        r += 1

    for col, w in [(1, 32), (2, 20)]:
        _w(ws1, col, w)
    ws1.freeze_panes = "A3"

    VS_set = set(inst.VS)

    # SHEET 2: Phase 1 Detail
    ws2 = wb.create_sheet("Phase 1 Detail")
    title(ws2, 1, "Phase 1 - No-Disruption MIP Facility Decisions", 5)
    hdr_row(ws2, 2, ["Facility", "Type", "MIP decision y",
                     "Fixed cost", "Open in final?"], bg=C_LIGHT)

    y_start = history_p1["y_start"]

    for ri, j in enumerate(
            sorted(inst.V0, key=lambda jj: -y_start.get(jj, 0)), 3):
        mip_val = y_start.get(j, 0)
        final = y_star.get(j, 0)
        ftype = "Supply" if j in VS_set else "Transshipment"
        bg = C_GREEN if mip_val == 1 else C_GRAY

        cell(ws2, ri, 1, j, bg=bg)
        cell(ws2, ri, 2, ftype, bg=bg)
        cell(ws2, ri, 3, "1 (open)" if mip_val else "0 (closed)", bg=bg)
        cell(ws2, ri, 4, round(inst.f[j], 2), bg=bg, fmt="#,##0.00")
        cell(ws2, ri, 5, "OPEN" if final else "closed",
             bg=C_GREEN if final else C_WHITE,
             color="375623" if final else "888888")

    for col, w in [(1, 10), (2, 16), (3, 16), (4, 14), (5, 14)]:
        _w(ws2, col, w)

    ws2.freeze_panes = "A3"

    # SHEET 3: Phase 2 NS log
    ws3 = wb.create_sheet("Phase 2 NS Log")
    title(ws3, 1, "Phase 2 - Neighborhood Search Improvement Log", 12)

    if history_p2:
        hdr_row(ws3, 2,
                ["Iter", "Sweep", "Move type",
                 "UB after", "Gap %", "UB improvement",
                 "Recourse cost", "Disrupted facilities",
                 "SubP (s)", "Warm LP skip",
                 "Candidates", "Pool size"],
                bg=C_LIGHT)

        prev_UB = history_p1["UB_start"]
        for ri, h in enumerate(history_p2, 3):
            improv = prev_UB - h["UB"]
            bg = C_STAR if improv > 0 else C_WHITE

            z_hat = h.get("z_hat", {})
            dis_list = sorted(j for j in inst.V0 if z_hat.get(j, 0) == 1)
            dis_str = str(dis_list) if dis_list else "none"

            y_h = h.get("y_hat", {})
            fc_h = sum(inst.f[j] * y_h.get(j, 0) for j in inst.V0) if y_h else 0.0
            rc_h = round(h["UB"] - fc_h, 2) if y_h else "-"

            vals = [
                h["iter"],
                h["sweep"],
                h["move_type"],
                round(h["UB"], 2),
                round(h["gap"], 4),
                round(improv, 2),
                rc_h,
                dis_str,
                round(h["subp_time"], 3),
                h.get("n_lp_skip", ""),
                h.get("n_candidates", ""),
                h.get("pool_size", ""),
            ]
            fmts = [None, None, None, "#,##0.00", "0.0000", "#,##0.00",
                    "#,##0.00", None, "0.000", None, None, None]
            for ci, (v, fmt) in enumerate(zip(vals, fmts), 1):
                cell(ws3, ri, ci, v, bg=bg, fmt=fmt,
                     align="left" if ci in (3, 8) else "center")
            ws3.row_dimensions[ri].height = 16
            prev_UB = h["UB"]

        tot = len(history_p2) + 3
        cell(ws3, tot, 1, "TOTAL", bg=C_LIGHT, bold=True)
        cell(ws3, tot, 6,
             round(history_p1["UB_start"] - UB, 2),
             bg=C_STAR, bold=True, fmt="#,##0.00")
        cell(ws3, tot, 9,
             round(sum(h["subp_time"] for h in history_p2), 3),
             bg=C_LIGHT, bold=True, fmt="0.000")
    else:
        ws3.cell(row=3, column=1,
                 value="No improvements found in Phase 2.")

    for col, w in [(1, 6), (2, 7), (3, 28), (4, 14), (5, 10), (6, 16),
                   (7, 14), (8, 22), (9, 10), (10, 13), (11, 11), (12, 10)]:
        _w(ws3, col, w)
    ws3.freeze_panes = "A3"

    # SHEET 4: Facility decisions
    ws4 = wb.create_sheet("Facility Decisions")
    title(ws4, 1,
          "Facility Decisions & Worst-Case Disruption - Final Solution", 7)

    if history_p2:
        z_final = history_p2[-1].get("z_hat", history_p1["z_init"])
    else:
        z_final = history_p1["z_init"]

    hdr_row(ws4, 2,
            ["Facility", "Type", "Fixed cost", "Capacity",
             "Status", "MIP start y", "Disrupted?"],
            bg=C_LIGHT)

    for ri, j in enumerate(sorted(inst.V0), 3):
        is_open = y_star.get(j, 0) == 1
        is_dis = z_final.get(j, 0) == 1
        ftype = "Supply" if j in VS_set else "Transshipment"

        if is_dis and is_open:
            bg = C_RED
        elif is_open:
            bg = C_GREEN
        else:
            bg = C_GRAY

        dis_label = "DISRUPTED" if is_dis else ("-" if not is_open else "safe")

        vals = [
            j,
            ftype,
            round(inst.f[j], 2),
            round(inst.Q[j], 2),
            "OPEN" if is_open else "closed",
            round(history_p1["y_start"].get(j, 0), 4),
            dis_label
        ]
        fmts = [None, None, "#,##0.00", "#,##0.00", None, "0.0000", None]

        for ci, (v, fmt) in enumerate(zip(vals, fmts), 1):
            dis_color = "CC0000" if (is_dis and ci == 7) else (
                        "375623" if (is_open and ci == 5) else "000000")
            cell(ws4, ri, ci, v,
                 bg=bg,
                 fmt=fmt,
                 color=dis_color,
                 bold=(is_dis and ci == 7))
        ws4.row_dimensions[ri].height = 15

    sr = len(inst.V0) + 4
    section(ws4, sr, "WORST-CASE DISRUPTION SUMMARY", 7)
    sr += 1

    dis_facs = [j for j in inst.V0
                if z_final.get(j, 0) == 1 and y_star.get(j, 0) == 1]
    cap_lost = sum(inst.Q[j] for j in dis_facs)

    dis_summary = [
        ("Disrupted facilities", str(sorted(dis_facs))),
        ("Number disrupted",     len(dis_facs)),
        ("Total capacity lost",  round(cap_lost, 2)),
    ]
    for lbl, val in dis_summary:
        bg = C_RED if "lost" in lbl or "Disrupted" in lbl else C_GRAY
        cell(ws4, sr, 1, lbl, bg=bg, bold=True, align="left")
        cell(ws4, sr, 2, val, bg=bg,
             fmt="#,##0.00" if isinstance(val, float) else None)
        sr += 1

    for col, w in [(1, 10), (2, 16), (3, 14), (4, 12),
                   (5, 10), (6, 14), (7, 16)]:
        _w(ws4, col, w)
    ws4.freeze_panes = "A3"

    # SHEET 5: Flow details
    ws5 = wb.create_sheet("Flow Details")
    title(ws5, 1,
          "Flow Details - Final Solution Under Worst-Case Disruption", 6)

    print("  [Excel] Solving recourse flows for final solution...")
    flows, unmet = _solve_flows(inst, y_star, z_final)

    section(ws5, 2, "A. ARC FLOWS (non-zero only)", 6)
    hdr_row(ws5, 3,
            ["From", "To", "Arc type", "Arc cost", "Flow", "Flow x cost"],
            bg=C_LIGHT)

    VS_s = set(inst.VS); VT_s = set(inst.VT); VD_s = set(inst.VD)
    ri = 4
    total_transport = 0.0
    for (i, j) in sorted(inst.arcs):
        flow = flows.get((i, j), 0.0)
        if flow < 1e-4:
            continue
        if i in VS_s and j in VT_s:
            arc_type = "S-T"
        elif i in VS_s:
            arc_type = "S-D"
        else:
            arc_type = "T-D"
        cost = inst.c[(i, j)]
        flow_cost = flow * cost
        total_transport += flow_cost
        bg = C_GRAY if ri % 2 == 0 else C_WHITE
        cell(ws5, ri, 1, i,                   bg=bg)
        cell(ws5, ri, 2, j,                   bg=bg)
        cell(ws5, ri, 3, arc_type,            bg=bg)
        cell(ws5, ri, 4, round(cost, 2),      bg=bg, fmt="#,##0.00")
        cell(ws5, ri, 5, round(flow, 4),      bg=bg, fmt="#,##0.00")
        cell(ws5, ri, 6, round(flow_cost, 2), bg=bg, fmt="#,##0.00")
        ws5.row_dimensions[ri].height = 14
        ri += 1

    cell(ws5, ri, 1, "Total transport cost", bg=C_LIGHT, bold=True,
         align="left")
    cell(ws5, ri, 6, round(total_transport, 2),
         bg=C_STAR, bold=True, fmt="#,##0.00")
    ws5.merge_cells(start_row=ri, start_column=1, end_row=ri, end_column=5)
    ri += 2

    section(ws5, ri, "B. DEMAND NODE FULFILMENT", 6)
    ri += 1
    hdr_row(ws5, ri,
            ["Demand node", "Required demand", "Flow received",
             "Unmet demand", "Unmet %", "Penalty cost"],
            bg=C_LIGHT)
    ri += 1
    total_penalty = 0.0
    for d in sorted(inst.VD):
        required = -inst.b[d]
        received = sum(flows.get((i, d), 0.0) for i in inst.Vminus[d])
        um = unmet.get(d, 0.0)
        pct = um / required * 100 if required > 0 else 0.0
        penalty = um * inst.theta[d]
        total_penalty += penalty
        bg = C_RED if um > 1e-4 else (C_GRAY if ri % 2 == 0 else C_WHITE)
        cell(ws5, ri, 1, d,                   bg=bg)
        cell(ws5, ri, 2, round(required, 4),  bg=bg, fmt="#,##0.00")
        cell(ws5, ri, 3, round(received, 4),  bg=bg, fmt="#,##0.00")
        cell(ws5, ri, 4, round(um, 4),        bg=bg, fmt="#,##0.00")
        cell(ws5, ri, 5, round(pct, 2),       bg=bg, fmt='0.00"%"')
        cell(ws5, ri, 6, round(penalty, 2),   bg=bg, fmt="#,##0.00")
        ws5.row_dimensions[ri].height = 14
        ri += 1

    cell(ws5, ri, 1, "Total penalty cost", bg=C_LIGHT, bold=True,
         align="left")
    cell(ws5, ri, 6, round(total_penalty, 2),
         bg=C_STAR, bold=True, fmt="#,##0.00")
    ws5.merge_cells(start_row=ri, start_column=1, end_row=ri, end_column=5)
    ri += 1
    cell(ws5, ri, 1, "Total recourse cost (transport + penalty)",
         bg=C_LIGHT, bold=True, align="left")
    cell(ws5, ri, 6, round(total_transport + total_penalty, 2),
         bg=C_STAR, bold=True, fmt="#,##0.00")
    ws5.merge_cells(start_row=ri, start_column=1, end_row=ri, end_column=5)

    for col, w in [(1, 14), (2, 18), (3, 16), (4, 14), (5, 10), (6, 16)]:
        _w(ws5, col, w)
    ws5.freeze_panes = "A3"

    wb.save(out_path)
    print(f"\n  Excel saved -> {out_path}")
    print(f"  Sheets: {[s.title for s in wb.worksheets]}")


# MAIN

def main():
    print("=" * 65)
    print("  RO-LNDP0  H2 (no-disruption MIP + Phase 2)")
    print("=" * 65)

    GEN_PARAMS = dict(
        nS       = 10,
        nT       = 20,
        nD       = 30,
        density  = 0.20,
        seed     = 4,
        m_budget = 5,
    )

    PHASE2_TIME_LIMIT = 10800.0
    POOL_CAP          = 0

    print(f"\nInstance           : {GEN_PARAMS}")
    print(f"Phase 2 time limit : {PHASE2_TIME_LIMIT}s")
    print(f"Pool cap           : {POOL_CAP if POOL_CAP > 0 else 'unlimited'}")

    raw = generate_instance_paper_exact(**GEN_PARAMS)
    summarize_instance(raw)
    inst = LNDPInstance.from_generator_dict(raw)

    print(f"\nNetwork: {len(inst.V)} nodes | {len(inst.arcs)} arcs | "
          f"{len(inst.V0)} facility nodes | m={inst.m}")

    t0 = time.time()
    y_star, UB, LB, history_p1, history_p2, status = h2_mip_ns(
        inst,
        phase2_time_limit = PHASE2_TIME_LIMIT,
        pool_cap          = POOL_CAP,
        verbose           = True,
    )
    total_time = time.time() - t0

    GEN_PARAMS["phase2_time_limit"] = PHASE2_TIME_LIMIT
    GEN_PARAMS["pool_cap"]          = POOL_CAP
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (f"{int(GEN_PARAMS['density']*100)}pct-"
             f"{GEN_PARAMS['nS']}-{GEN_PARAMS['nT']}-{GEN_PARAMS['nD']}")
    out_path = (f"H2_{label}_seed{GEN_PARAMS['seed']}"
                f"_m{GEN_PARAMS['m_budget']}_{ts}.xlsx")

    export_h2_excel(
        inst       = inst,
        raw        = raw,
        gen_params = GEN_PARAMS,
        history_p1 = history_p1,
        history_p2 = history_p2,
        y_star     = y_star,
        UB         = UB,
        LB         = LB,
        status     = status,
        total_time = total_time,
        out_path   = out_path,
    )


if __name__ == "__main__":
    main()
