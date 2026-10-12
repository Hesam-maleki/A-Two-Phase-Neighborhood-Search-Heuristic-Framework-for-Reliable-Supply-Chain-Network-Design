"""
phase2_filter_ns.py: Phase 2 shared by H1 and H2. Neighborhood search over
SWAP, ADD and DROP moves, screened by a reusable filter LP that is re-solved
with a warm dual-simplex start, and verified by the worst-case subproblem.
"""

import math
import time

from docplex.mp.model import Model

from ro_lndp0_ccg import LNDPInstance, solve_subproblem


class FilterLP:
    """
    Persistent LP encoding the accumulated worst-case scenarios.

    Built once per sweep from the current solution. Each neighbour is
    evaluated by updating only the capacity-constraint RHS values that
    changed and re-solving via dual-simplex warm restart. The filter
    objective is a valid lower bound on the true worst-case cost, so
    any neighbour with filter_cost >= UB can be safely skipped.

    pool_cap > 0 keeps only the most-recent pool_cap scenarios in the
    LP so its size stays bounded; pool_cap = 0 uses all scenarios.
    """

    @staticmethod
    def _select_scenarios(scenarios_list: list, pool_cap: int) -> list:
        """Return the most-recent pool_cap scenarios (all if pool_cap=0)."""
        if pool_cap > 0 and len(scenarios_list) > pool_cap:
            return scenarios_list[-pool_cap:]
        return scenarios_list

    def __init__(self, inst: LNDPInstance,
                 y_current: dict,
                 scenarios_list: list,
                 pool_cap: int = 0):
        self.inst = inst
        self.y_current = dict(y_current)
        self.scenarios_list = scenarios_list
        self.pool_cap = pool_cap
        self._active = self._select_scenarios(scenarios_list, pool_cap)
        self._model = None
        self._phi = None
        self._cap_ct_names = {}
        self._build()

    def _cap_ub(self, l: int, j: int, y: dict) -> float:
        z = self._active[l]
        return (1 - z.get(j, 0)) * self.inst.Q[j] * y[j]

    def _build(self):
        inst = self.inst
        m = Model(name="filter_lp", log_output=False)
        m.parameters.lpmethod.set(2)   # dual simplex (required for warm restart)
        phi = m.continuous_var(lb=0, name="phi")
        self._phi = phi

        for l, z_hat in enumerate(self._active):
            xl = {(i, j): m.continuous_var(lb=0, name=f"x{l}_{i}_{j}")
                  for (i, j) in inst.arcs}
            ul = {j: m.continuous_var(lb=0, name=f"u{l}_{j}")
                  for j in inst.VD}

            m.add_constraint(
                phi >= (m.sum(inst.c[(i, j)] * xl[(i, j)] for (i, j) in inst.arcs)
                      + m.sum(inst.theta[j] * ul[j] for j in inst.VD)),
                ctname=f"phi_{l}")

            for j in inst.VS:
                m.add_constraint(m.sum(xl[(j, k)] for k in inst.Vplus[j])
                                 <= inst.b[j], ctname=f"sup{l}_{j}")
            for j in inst.VT:
                m.add_constraint(m.sum(xl[(i, j)] for i in inst.Vminus[j]) ==
                                 m.sum(xl[(j, k)] for k in inst.Vplus[j]),
                                 ctname=f"tra{l}_{j}")
            for j in inst.VD:
                m.add_constraint(m.sum(xl[(i, j)] for i in inst.Vminus[j])
                                 + ul[j] == -inst.b[j],
                                 ctname=f"dem{l}_{j}")
            for j in inst.V0:
                ct_name = f"cap{l}_{j}"
                m.add_constraint(m.sum(xl[(j, k)] for k in inst.Vplus[j])
                                 <= self._cap_ub(l, j, self.y_current),
                                 ctname=ct_name)
                self._cap_ct_names[(l, j)] = ct_name

        m.minimize(phi)
        m.solve(log_output=False)
        self._model = m
        self._n_sc = len(self._active)

    def evaluate(self, y_candidate: dict, changed_facilities: list) -> float:
        """
        Re-solve for a neighbour via warm dual-simplex restart.

        Returns fixed_cost(y_candidate) + filter_LP_phi_value, which is
        a valid lower bound on the true worst-case total cost.
        """
        m = self._model

        for j in changed_facilities:
            for l in range(self._n_sc):
                ct = m.get_constraint_by_name(self._cap_ct_names[(l, j)])
                ct.rhs = self._cap_ub(l, j, y_candidate)

        sol = m.solve(log_output=False)
        phi_val = self._phi.solution_value if sol else math.inf

        for j in changed_facilities:
            for l in range(self._n_sc):
                ct = m.get_constraint_by_name(self._cap_ct_names[(l, j)])
                ct.rhs = self._cap_ub(l, j, self.y_current)
        m.solve(log_output=False)

        fc = sum(self.inst.f[j] * y_candidate[j] for j in self.inst.V0)
        return fc + phi_val

    def end(self):
        if self._model is not None:
            self._model.end()
            self._model = None


def _get_neighbors(y_current: dict, inst: LNDPInstance):
    """
    Yield (move_type, y_neighbor, description, changed_facilities).
    DROP is enabled only when the open count is at least m + 2.
    """
    open_facs = [j for j in inst.V0 if y_current[j] == 1]
    closed_facs = [j for j in inst.V0 if y_current[j] == 0]

    for cj in open_facs:
        for ok in closed_facs:
            yn = dict(y_current); yn[cj] = 0; yn[ok] = 1
            yield ("SWAP", yn, f"SWAP close={cj} open={ok}", [cj, ok])

    for ok in closed_facs:
        yn = dict(y_current); yn[ok] = 1
        yield ("ADD", yn, f"ADD open={ok}", [ok])

    if len(open_facs) >= inst.m + 2:
        for cj in open_facs:
            yn = dict(y_current); yn[cj] = 0
            yield ("DROP", yn, f"DROP close={cj}", [cj])


def run_phase2(inst: LNDPInstance,
               big_M: dict,
               y_init: dict,
               UB_init: float,
               LB_fixed: float,
               scenarios_set: set,
               scenarios_list: list,
               time_limit: float = 10800.0,
               pool_cap: int = 0,
               iter_offset: int = 0,
               verbose: bool = True):
    """
    Neighborhood search accelerated by a persistent warm-basis filter LP.

    Each sweep builds the filter LP once, evaluates every neighbour via
    warm dual-simplex, sorts the survivors by filter cost, and accepts
    the first candidate whose exact full cost improves UB. The search
    stops when a sweep finds no improving candidate.

    pool_cap caps the number of scenarios used in the filter LP per
    sweep (most recent kept); pool_cap = 0 means unlimited. iter_offset
    is added to the iteration number of each history entry.

    Returns
    -------
    y_best, UB, history, status
    """
    t_start = time.time()
    UB = UB_init
    y_current = dict(y_init)
    history = []
    status = "LOCAL_OPT"
    sweep_num = 0

    if verbose:
        print("\n" + "=" * 72)
        print("  PHASE 2 - Neighborhood Search  [Warm-Basis LP Filter]")
        print(f"  Starting UB : {UB_init:,.2f}  "
              f"Frozen LB : {LB_fixed:,.2f}  "
              f"pool_cap : {pool_cap if pool_cap > 0 else 'unlimited'}")
        print(f"  Scenarios from Phase 1 : {len(scenarios_list)}  "
              f"Time budget : {time_limit:.0f}s")
        print("=" * 72)

    while True:
        sweep_num += 1
        elapsed = time.time() - t_start
        if elapsed >= time_limit:
            status = "TIME_LIMIT"
            if verbose:
                print(f"\n  [Phase 2 time limit: {elapsed:.1f}s]")
            break

        open_now = sum(1 for j in inst.V0 if y_current[j] == 1)
        gap_now = ((UB - LB_fixed) / abs(UB) * 100
                   if UB > 0 else math.inf)

        if verbose:
            print(f"\n  Sweep {sweep_num}  |  UB={UB:,.2f}  "
                  f"gap={gap_now:.4f}%  open={open_now}  "
                  f"pool={len(scenarios_list)}")

        t_build = time.time()
        flp = FilterLP(inst, y_current, scenarios_list, pool_cap)
        build_t = time.time() - t_build

        if verbose:
            print(f"  FilterLP: {len(flp._active)} scenarios  "
                  f"built in {build_t:.3f}s")
            print(f"  {'#':>6}  {'Type':>5}  {'FilterCost':>14}  "
                  f"{'vs UB':>10}  {'pass':>5}")
            print("  " + "-" * 46)

        n_evaluated = 0
        n_lp_skip = 0
        candidates = []

        for move_type, y_neigh, desc, changed in \
                _get_neighbors(y_current, inst):

            elapsed = time.time() - t_start
            if elapsed >= time_limit:
                status = "TIME_LIMIT"; break

            n_evaluated += 1

            filt_cost = flp.evaluate(y_neigh, changed)
            delta = filt_cost - UB
            passed = filt_cost < UB - 1e-4

            if verbose:
                print(f"  {n_evaluated:>6}  {move_type:>5}  "
                      f"{filt_cost:>14.2f}  {delta:>+10.2f}  "
                      f"{'YES' if passed else '':>5}")

            if passed:
                candidates.append((filt_cost, dict(y_neigh), desc))
            else:
                n_lp_skip += 1

        flp.end()
        candidates.sort(key=lambda c: c[0])
        n_candidates = len(candidates)

        if verbose:
            print(f"\n  Sweep {sweep_num} summary:")
            print(f"    Neighbours evaluated  : {n_evaluated}")
            print(f"    Skipped by filter LP  : {n_lp_skip}")
            print(f"    Passed to subproblem  : {n_candidates}")

        if not candidates or status == "TIME_LIMIT":
            if status != "TIME_LIMIT":
                status = "LOCAL_OPT"
                if verbose:
                    print(f"\n  No candidate passed filter -> local optimum")
            break

        accepted = None
        time_limit_hit = False
        subp_total = 0.0
        n_verified = 0

        for rank, (filt_cost, cand_y, cand_desc) in enumerate(candidates):
            elapsed = time.time() - t_start
            if elapsed >= time_limit:
                status = "TIME_LIMIT"
                time_limit_hit = True
                break

            if verbose:
                print(f"\n  Verify #{rank+1}/{n_candidates}: {cand_desc}  "
                      f"filter={filt_cost:,.2f}")

            remaining = max(10.0, time_limit - (time.time() - t_start))
            t_sub = time.time()
            z_new, chi = solve_subproblem(inst, cand_y, big_M, remaining)
            subp_time = time.time() - t_sub
            subp_total += subp_time
            n_verified += 1

            fc_new = sum(inst.f[j] * cand_y[j] for j in inst.V0)
            full_cost = fc_new + chi
            gap_new = ((full_cost - LB_fixed) / abs(full_cost) * 100
                       if full_cost > 0 else math.inf)

            z_key = frozenset(j for j in inst.V0 if z_new.get(j, 0) == 1)
            new_sc = z_key not in scenarios_set
            if new_sc:
                scenarios_set.add(z_key)
                scenarios_list.append(dict(z_new))

            err = full_cost - filt_cost
            improves = full_cost < UB - 1e-6

            if verbose:
                marker = "  accepted" if improves else ""
                print(f"    Full={full_cost:,.2f}  filter={filt_cost:,.2f}  "
                      f"error={err:+.2f}  subp={subp_time:.3f}s"
                      f"{'  [new scenario]' if new_sc else ''}{marker}")

            if improves:
                accepted = (full_cost, dict(cand_y), cand_desc,
                            dict(z_new), subp_time, gap_new,
                            rank + 1, new_sc)
                break

        if accepted is None:
            if not time_limit_hit:
                if verbose:
                    print(f"\n  All {n_candidates} candidates verified, "
                          f"no improvement -> local optimum")
                status = "LOCAL_OPT"
            break

        (acc_cost, acc_y, acc_desc, acc_z,
         acc_subp, acc_gap, acc_rank, acc_new_sc) = accepted

        UB = acc_cost
        y_current = acc_y
        elapsed = time.time() - t_start
        n_skipped = n_candidates - n_verified

        history.append(dict(
            iter          = iter_offset + len(history) + 1,
            sweep         = sweep_num,
            LB            = LB_fixed,
            UB            = UB,
            gap           = acc_gap,
            time_elapsed  = elapsed,
            subp_time     = acc_subp,
            master_time   = 0.0,
            iter_time     = elapsed,
            build_time    = build_t,
            subp_status   = "optimal",
            master_status = "warm_filter",
            z_hat         = acc_z,
            y_hat         = dict(y_current),
            phase         = 2,
            move_type     = acc_desc,
            n_evaluated   = n_evaluated,
            n_lp_skip     = n_lp_skip,
            n_candidates  = n_candidates,
            n_verified    = n_verified,
            n_skipped     = n_skipped,
            rank_accepted = acc_rank,
            pool_size     = len(scenarios_list),
            sweep_timing  = dict(
                sweep            = sweep_num,
                n_evaluated      = n_evaluated,
                n_skipped_lp     = n_lp_skip,
                n_passed_filter  = n_candidates,
                n_verified       = n_verified,
                n_skipped_first  = n_skipped,
                filter_rate_pct  = round(
                    n_lp_skip / max(n_evaluated, 1) * 100, 1),
                sweep_time_s     = round(elapsed, 3),
                t_build_filter_s = round(build_t, 3),
                t_subp_total_s   = round(subp_total, 3),
                avg_subp_s       = round(
                    subp_total / max(n_verified, 1), 3),
                n_scenarios      = len(scenarios_list),
            ),
        ))

        if verbose:
            oc = sum(1 for j in inst.V0 if y_current[j] == 1)
            print(f"\n  UB -> {UB:,.2f}  gap={acc_gap:.4f}%  |  "
                  f"{acc_desc}  (accepted at rank {acc_rank}/{n_candidates})  "
                  f"open={oc}")
            print(f"     verified={n_verified}  skipped={n_skipped}  "
                  f"subp_total={subp_total:.2f}s")

        if time_limit_hit:
            break

    elapsed = time.time() - t_start
    gap_f = ((UB - LB_fixed) / abs(UB) * 100 if UB > 0 else math.inf)
    if verbose:
        print(f"\n  Phase 2 finished  ({elapsed:.1f}s  "
              f"status={status}  UB={UB:,.2f}  gap={gap_f:.4f}%)")

    return y_current, UB, history, status
