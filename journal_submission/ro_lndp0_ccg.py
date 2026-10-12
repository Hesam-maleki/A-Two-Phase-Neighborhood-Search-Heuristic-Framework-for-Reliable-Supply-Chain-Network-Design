"""
ro_lndp0_ccg.py: RO-LNDP0 model components (instance class, big-M values,
linearised subproblem, master problem, C&CG cut) and the exact
column-and-constraint generation algorithm with LP enhancement
(Cheng et al., 2018).
"""

import math
import time
from docplex.mp.model import Model

from instance_generator_paper_exact import (
    generate_instance_paper_exact,
    summarize_instance,
)


# INTERNAL DATA STRUCTURE

class LNDPInstance:
    """
    Three-echelon LNDP instance.

    Attributes
    ----------
    VS, VT, VD  list   supply / transshipment / demand node indices
    V0          list   facility nodes = VS u VT
    V           list   all nodes
    f           dict   {j: float}       fixed opening cost,   j in V0
    Q           dict   {j: float}       capacity,             j in V0
    b           dict   {j: float}       supply (>=0) or demand (<=0)
    theta       dict   {j: float}       unmet-demand penalty, j in VD
    arcs        list   [(i,j)]          directed arcs
    c           dict   {(i,j): float}   unit transportation cost
    Vplus       dict   {j: list}        successors   (out-arcs of j)
    Vminus      dict   {j: list}        predecessors (in-arcs  of j)
    m           int    disruption budget
    """

    def __init__(self, VS, VT, VD, f, Q, b, theta, arcs, c, m):
        self.VS    = list(VS)
        self.VT    = list(VT)
        self.VD    = list(VD)
        self.V0    = self.VS + self.VT
        self.V     = self.V0 + self.VD
        self.f     = f
        self.Q     = Q
        self.b     = b
        self.theta = theta
        self.arcs  = list(arcs)
        self.c     = c
        self.m     = m

        self.Vplus  = {j: [] for j in self.V}
        self.Vminus = {j: [] for j in self.V}
        for (i, j) in self.arcs:
            self.Vplus[i].append(j)
            self.Vminus[j].append(i)

    @classmethod
    def from_generator_dict(cls, d):
        """Convert the dict from generate_instance_paper_exact() to LNDPInstance."""
        arcs = list(d["A_cost"].keys())
        c    = dict(d["A_cost"])
        return cls(
            VS    = d["VS"],
            VT    = d["VT"],
            VD    = d["VD"],
            f     = {int(k): v for k, v in d["f"].items()},
            Q     = {int(k): v for k, v in d["Q"].items()},
            b     = {int(k): v for k, v in d["b"].items()},
            theta = {int(k): v for k, v in d["theta"].items()},
            arcs  = arcs,
            c     = c,
            m     = int(d["m_budget"]),
        )


# BIG-M COMPUTATION  (Section 4.1.1, Eq. 32)

def compute_big_M(inst):
    """
    Compute M_j for each facility j in V0.

    M_j = max( max_{i in VD} { theta_i - c*_{ji} },
               max_{k in V+_j} { c_{jk} } )

    c*_{ji} = min( c_{ji},  min_{k in VT}{ c_{jk} + c_{ki} } )
    """
    INF = 1e12
    M   = {}
    for j in inst.V0:
        M1 = 0.0
        for i in inst.VD:
            direct   = inst.c.get((j, i), INF)
            indirect = min(
                (inst.c.get((j, k), INF) + inst.c.get((k, i), INF)
                 for k in inst.VT),
                default=INF,
            )
            M1 = max(M1, inst.theta[i] - min(direct, indirect))
        M2   = max((inst.c.get((j, k), 0.0) for k in inst.Vplus[j]), default=0.0)
        M[j] = max(M1, M2)
    return M


# LINEARISED SUBPROBLEM  L-SubP  (Section 4.1.1, Eqs. 21-33)

def solve_subproblem(inst, y_hat, big_M, time_limit=300.0):
    """
    Solve the linearised max-max subproblem for a fixed y_hat.

    The primal second-stage min problem is dualised to a maximisation over
    dual variables (alpha, beta, gamma, pi).  The bilinear term z_j * pi_j
    is linearised via  w_j = z_j * pi_j  with big-M constraints (Eqs. 28-31).

    Returns
    -------
    z_hat : dict {j: 0/1}   worst-case disruption scenario
    chi   : float            worst-case recourse cost
    """
    sp = Model(name="L_SubP")
    sp.parameters.timelimit = time_limit

    VS_set = set(inst.VS)
    VT_set = set(inst.VT)
    VD_set = set(inst.VD)

    # Dual variables
    alpha = {j: sp.continuous_var(lb=-sp.infinity, ub=0,            name=f"al_{j}") for j in inst.VS}
    beta  = {j: sp.continuous_var(lb=-sp.infinity,                  name=f"be_{j}") for j in inst.VT}
    gamma = {j: sp.continuous_var(lb=-sp.infinity, ub=inst.theta[j],name=f"ga_{j}") for j in inst.VD}
    pi    = {j: sp.continuous_var(lb=-sp.infinity, ub=0,            name=f"pi_{j}") for j in inst.V0}

    # Binary disruption variables
    z = {j: sp.binary_var(name=f"z_{j}") for j in inst.V0}

    # Linearisation  w_j = z_j * pi_j  (Eqs. 28-31)
    Mj = big_M
    w  = {j: sp.continuous_var(lb=-Mj[j], ub=0, name=f"w_{j}") for j in inst.V0}
    for j in inst.V0:
        sp.add_constraint(w[j] >= pi[j])                         # Eq. 28
        sp.add_constraint(w[j] >= -Mj[j] * z[j])                # Eq. 29
        sp.add_constraint(w[j] <= pi[j] + Mj[j] * (1 - z[j]))  # Eq. 30
        # Eq. 31: w[j] <= 0 already enforced by ub=0

    # Budget constraint  sum z_j <= m  (Eq. 1)
    sp.add_constraint(sp.sum(z[j] for j in inst.V0) <= inst.m)

    # Dual feasibility constraints (Eqs. 22-24)
    for (i, j) in inst.arcs:
        if i in VS_set and j in VT_set:
            sp.add_constraint(alpha[i] + beta[j]  + pi[i] <= inst.c[(i, j)])
        elif i in VS_set and j in VD_set:
            sp.add_constraint(alpha[i] + gamma[j] + pi[i] <= inst.c[(i, j)])
        elif i in VT_set and j in VD_set:
            sp.add_constraint(-beta[i] + gamma[j] + pi[i] <= inst.c[(i, j)])

    # Objective (Eq. 33)
    obj = (sp.sum(inst.b[j] * alpha[j]                    for j in inst.VS)
         - sp.sum(inst.b[j] * gamma[j]                    for j in inst.VD)
         + sp.sum(inst.Q[j] * y_hat[j] * (pi[j] - w[j])  for j in inst.V0))
    sp.maximize(obj)

    sol   = sp.solve(log_output=False)
    chi   = sp.objective_value if sol else 0.0
    z_hat = ({j: int(round(z[j].solution_value)) for j in inst.V0}
             if sol else {j: 0 for j in inst.V0})
    sp.end()
    return z_hat, chi


# MASTER PROBLEM  (Section 4.1.2, Eqs. 34-42)

def build_master(inst):
    """Build the initial empty master problem (no C&CG cuts yet)."""
    mp  = Model(name="MP")
    y   = {j: mp.binary_var(name=f"y_{j}") for j in inst.V0}
    phi = mp.continuous_var(lb=0, name="phi")
    # Eq. 34: min sum f_j y_j + phi
    mp.minimize(mp.sum(inst.f[j] * y[j] for j in inst.V0) + phi)
    return mp, y, phi


def add_ccg_cut(mp, inst, y, phi, z_hat, scenario_idx):
    """
    Add one C&CG cut for worst-case scenario z_hat (iteration l).

    Introduces recourse variables x^l, u^l and the constraints
    Eqs. 35-42 for this scenario.
    """
    l   = scenario_idx
    x_l = {(i, j): mp.continuous_var(lb=0, name=f"x{l}_{i}_{j}")
           for (i, j) in inst.arcs}
    u_l = {j: mp.continuous_var(lb=0, name=f"u{l}_{j}")
           for j in inst.VD}

    # Eq. 35: phi >= sum c_ij x^l_ij + sum theta_j u^l_j
    mp.add_constraint(
        phi >= (mp.sum(inst.c[(i, j)] * x_l[(i, j)] for (i, j) in inst.arcs)
              + mp.sum(inst.theta[j]  * u_l[j]       for j in inst.VD)))

    # Eq. 36: supply flow conservation
    for j in inst.VS:
        mp.add_constraint(
            mp.sum(x_l[(j, k)] for k in inst.Vplus[j]) <= inst.b[j])

    # Eq. 37: transshipment flow conservation
    for j in inst.VT:
        mp.add_constraint(
            mp.sum(x_l[(i, j)] for i in inst.Vminus[j])
         == mp.sum(x_l[(j, k)] for k in inst.Vplus[j]))

    # Eq. 38: demand satisfaction
    for j in inst.VD:
        mp.add_constraint(
            mp.sum(x_l[(i, j)] for i in inst.Vminus[j]) + u_l[j] == -inst.b[j])

    # Eq. 39: capacity with disruption scenario z_hat
    for j in inst.V0:
        mp.add_constraint(
            mp.sum(x_l[(j, k)] for k in inst.Vplus[j])
            <= (1 - z_hat[j]) * inst.Q[j] * y[j])


# LP RELAXATION HELPER  (C&CG LP enhancement, Section 4.2)

def solve_mp_lp(mp, y):
    """
    Solve the LP relaxation of the master problem.

    Clones the model, relaxes all binary y_j to continuous [0,1],
    solves, then discards the clone.  Returns the LP optimal value
    (a valid lower bound on the MIP).
    """
    lp = mp.clone()
    lp.parameters.timelimit = 300
    for var in list(lp.iter_integer_vars()):
        lp.change_var_type(var, lp.continuous_vartype)
        lp.change_var_bounds(var, 0.0, 1.0)
    sol = lp.solve(log_output=False)
    lb  = lp.objective_value if sol else float("-inf")
    lp.end()
    return lb


# C&CG ALGORITHM  (Section 4.1.2 + LP enhancement)

def ccg_algorithm(inst,
                  eps=1e-6,
                  time_limit=10800.0,
                  use_lp_enhancement=True,
                  verbose=True):
    """
    Column-and-Constraint Generation algorithm for RO-LNDP0.

    Parameters
    ----------
    inst               LNDPInstance
    eps                float   relative gap tolerance
    time_limit         float   wall-clock seconds
    use_lp_enhancement bool    solve the LP relaxation of the master each iteration
    verbose            bool    print a per-iteration log

    Returns
    -------
    y_star  dict {j: 0/1}   best facility location decisions found
    UB      float            best upper bound  (worst-case total cost)
    LB      float            best lower bound
    history list of dicts    one dict per iteration with keys:
                               iter, LB, UB, gap, time_elapsed,
                               subp_time, master_time, iter_time,
                               subp_status, master_status, z_hat
    status  str              "OPTIMAL" or "TIME_LIMIT"
    """
    t_start = time.time()
    big_M   = compute_big_M(inst)

    LB      = -math.inf
    UB      =  math.inf
    y_star  = None
    history = []
    r       = 0
    status  = "OPTIMAL"

    mp, y, phi = build_master(inst)

    # Initial y_hat: open all facilities
    y_hat = {j: 1 for j in inst.V0}

    if verbose:
        hdr = (f"{'Iter':>4}  {'LB':>16}  {'UB':>16}  {'Gap%':>9}  "
               f"{'SubP(s)':>8}  {'MP(s)':>8}  {'Iter(s)':>8}")
        print("\n" + hdr)
        print("-" * len(hdr))

    while True:
        # Time limit check
        elapsed = time.time() - t_start
        if elapsed >= time_limit:
            status = "TIME_LIMIT"
            if verbose:
                print(f"  [Time limit reached: {elapsed:.1f}s]")
            break

        iter_t0 = time.time()

        # Subproblem: find worst-case disruption scenario
        remaining  = max(10.0, time_limit - elapsed)
        subp_t0    = time.time()
        z_hat, chi = solve_subproblem(inst, y_hat, big_M, remaining)
        subp_time  = time.time() - subp_t0
        r += 1

        # Add C&CG cut for this scenario
        add_ccg_cut(mp, inst, y, phi, z_hat, r)

        # Update UB
        fc_hat  = sum(inst.f[j] * y_hat[j] for j in inst.V0)
        UB_cand = fc_hat + chi
        if UB_cand < UB:
            UB     = UB_cand
            y_star = dict(y_hat)

        # Master problem: get new y_hat and tighter LB
        master_t0 = time.time()

        if use_lp_enhancement:
            lb_lp = solve_mp_lp(mp, y)
            LB    = max(LB, lb_lp)

        sol_mp        = mp.solve(log_output=False)
        master_time   = time.time() - master_t0
        master_status = "optimal" if sol_mp else "infeasible/no_sol"

        if sol_mp:
            LB    = max(LB, mp.objective_value)
            y_hat = {j: int(round(y[j].solution_value)) for j in inst.V0}

        # Gap check and logging
        gap       = (UB - LB) / abs(UB) * 100 if UB not in (0.0, math.inf) else math.inf
        elapsed   = time.time() - t_start
        iter_time = time.time() - iter_t0

        history.append(dict(
            iter          = r,
            LB            = LB,
            UB            = UB,
            gap           = gap,
            time_elapsed  = elapsed,
            subp_time     = subp_time,
            master_time   = master_time,
            iter_time     = iter_time,
            subp_status   = "optimal",
            master_status = master_status,
            z_hat         = dict(z_hat),
        ))

        if verbose:
            print(f"{r:>4}  {LB:>16.2f}  {UB:>16.2f}  {gap:>8.4f}%  "
                  f"{subp_time:>8.2f}  {master_time:>8.2f}  {iter_time:>8.2f}")

        if gap <= eps * 100:
            status = "OPTIMAL"
            if verbose:
                print(f"\n  Converged  (gap = {gap:.6f}%,  {r} iterations)")
            break

    mp.end()
    return y_star, UB, LB, history, status


# MAIN

def main():
    print("=" * 65)
    print("  RO-LNDP0  . Two-Stage Robust Logistics Network Design")
    print("  C&CG LP   . Cheng et al., TRB 2018")
    print("=" * 65)

    GEN_PARAMS = dict(
        nS       = 10,
        nT       = 20,
        nD       = 30,
        density  = 0.20,
        seed     = 4,
        m_budget = 5,
    )

    EPS        = 1e-6
    TIME_LIMIT = 10800.0

    print(f"\nInstance : {GEN_PARAMS}")
    raw  = generate_instance_paper_exact(**GEN_PARAMS)
    summarize_instance(raw)

    inst = LNDPInstance.from_generator_dict(raw)
    print(f"\nNetwork  : {len(inst.V)} nodes | {len(inst.arcs)} arcs | "
          f"{len(inst.V0)} facility nodes | m = {inst.m}")

    print("\n" + "-" * 65)
    print("  Solving RO-LNDP0 with C&CG LP")
    print("-" * 65)

    y_ro, UB_ro, LB_ro, hist, solve_status = ccg_algorithm(
        inst,
        eps                = EPS,
        time_limit         = TIME_LIMIT,
        use_lp_enhancement = True,
        verbose            = True,
    )

    open_ro = [j for j in inst.V0 if y_ro.get(j, 0) == 1]
    fc_ro   = sum(inst.f[j] * y_ro.get(j, 0) for j in inst.V0)
    gap_pct = (UB_ro - LB_ro) / abs(UB_ro) * 100

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Status              : {solve_status}")
    print(f"  Iterations          : {len(hist)}")
    print(f"  Upper bound (UB)    : {UB_ro:,.2f}")
    print(f"  Lower bound (LB)    : {LB_ro:,.2f}")
    print(f"  Optimality gap      : {gap_pct:.4f}%")
    print(f"  Fixed cost          : {fc_ro:,.2f}")
    print(f"  Worst-case recourse : {UB_ro - fc_ro:,.2f}")
    print(f"  Open facilities ({len(open_ro):>2}) : {open_ro}")
    print("=" * 65)


if __name__ == "__main__":
    main()