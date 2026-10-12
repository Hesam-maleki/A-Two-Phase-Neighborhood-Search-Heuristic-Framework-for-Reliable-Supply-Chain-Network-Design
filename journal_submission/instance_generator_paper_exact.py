"""
instance_generator_paper_exact.py: instance generator for RO-LNDP0 following
Cheng et al. (2018), Section 5.1.
"""

import random
import json
from pathlib import Path


def generate_instance_paper_exact(
    nS: int = 10,
    nT: int = 20,
    nD: int = 30,
    density: float = 0.3,
    seed: int = 1,
    theta_value: float = 1500.0,
    demand_low: float = 50.0,
    demand_high: float = 110.0,
    fixed_low: float = 5000.0,
    fixed_high: float = 15000.0,
    arc_cost_low: float = 1.0,
    arc_cost_high: float = 500.0,
    supply_cap_low_factor: float = 1.5,
    supply_cap_high_factor: float = 2.5,
    trans_cap_low_factor: float = 1.5,
    trans_cap_high_factor: float = 2.5,
    m_budget: int = 5,
    apply_connectivity_repair: bool = True,
) -> dict:
    """
    Generate one instance.

    Demands b_j ~ U(-110, -50); supply capacities ~ U(1.5 s_bar, 2.5 s_bar);
    transshipment capacities ~ U(1.5 c_bar, 2.5 c_bar); fixed costs
    ~ U(5000, 15000); each arc exists with probability `density` and has
    cost ~ U(1, 500); unmet-demand penalty theta = 1500.

    Returns a dict with keys VS, VT, VD, V0, V, A_cost, b, Q, f, theta,
    m_budget and meta.
    """
    rng = random.Random(seed)

    VS = list(range(1, nS + 1))
    VT = list(range(nS + 1, nS + nT + 1))
    VD = list(range(nS + nT + 1, nS + nT + nD + 1))
    V0 = VS + VT
    V  = V0 + VD

    b = {}
    for d in VD:
        b[d] = -rng.uniform(demand_low, demand_high)

    total_demand = sum(-b[d] for d in VD)

    s_bar = total_demand / max(1, nS)
    c_bar = total_demand / max(1, nT)

    Q = {}
    for s in VS:
        cap  = rng.uniform(supply_cap_low_factor * s_bar,
                           supply_cap_high_factor * s_bar)
        Q[s] = max(1.0, cap)
        b[s] = Q[s]

    for t in VT:
        cap  = rng.uniform(trans_cap_low_factor * c_bar,
                           trans_cap_high_factor * c_bar)
        Q[t] = max(1.0, cap)
        b[t] = 0.0

    f = {j: rng.uniform(fixed_low, fixed_high) for j in V0}

    theta = {d: float(theta_value) for d in VD}

    A_cost: dict[tuple, float] = {}

    for s in VS:
        for t in VT:
            if rng.random() <= density:
                A_cost[(s, t)] = rng.uniform(arc_cost_low, arc_cost_high)

    for s in VS:
        for d in VD:
            if rng.random() <= density:
                A_cost[(s, d)] = rng.uniform(arc_cost_low, arc_cost_high)

    for t in VT:
        for d in VD:
            if rng.random() <= density:
                A_cost[(t, d)] = rng.uniform(arc_cost_low, arc_cost_high)

    # Make sure every demand node has an incoming arc and every
    # transshipment node has an incoming and an outgoing arc.
    if apply_connectivity_repair:
        for d in VD:
            if not any((i, d) in A_cost for i in (VS + VT)):
                src = rng.choice(VT if VT else VS)
                A_cost[(src, d)] = rng.uniform(arc_cost_low, arc_cost_high)

        for t in VT:
            if not any((s, t) in A_cost for s in VS):
                s = rng.choice(VS)
                A_cost[(s, t)] = rng.uniform(arc_cost_low, arc_cost_high)
            if not any((t, d) in A_cost for d in VD):
                d = rng.choice(VD)
                A_cost[(t, d)] = rng.uniform(arc_cost_low, arc_cost_high)

    max_possible_arcs = nS * nT + nS * nD + nT * nD
    actual_density    = len(A_cost) / max_possible_arcs if max_possible_arcs else 0.0

    meta = {
        "generator_version"  : "cheng18",
        "seed"               : seed,
        "requested_density"  : density,
        "actual_density"     : round(actual_density, 6),
        "nS"                 : nS,
        "nT"                 : nT,
        "nD"                 : nD,
        "theta_value"        : theta_value,
        "demand_range"       : [demand_low, demand_high],
        "fixed_range"        : [fixed_low, fixed_high],
        "arc_cost_range"     : [arc_cost_low, arc_cost_high],
        "s_bar"              : round(s_bar, 4),
        "c_bar"              : round(c_bar, 4),
        "total_demand"       : round(total_demand, 4),
        "total_supply"       : round(sum(b[s] for s in VS), 4),
        "n_arcs"             : len(A_cost),
        "m_budget"           : int(m_budget),
        "connectivity_repair": apply_connectivity_repair,
    }

    return {
        "VS"      : VS,
        "VT"      : VT,
        "VD"      : VD,
        "V0"      : V0,
        "V"       : V,
        "A_cost"  : A_cost,
        "b"       : b,
        "Q"       : Q,
        "f"       : f,
        "theta"   : theta,
        "m_budget": int(m_budget),
        "meta"    : meta,
    }


def save_instance_json(inst: dict, filepath: str) -> None:
    """Save an instance to JSON; arc keys (i, j) are written as "i,j"."""
    payload = {k: v for k, v in inst.items() if k != "A_cost"}
    payload["A_cost"] = {f"{i},{j}": c for (i, j), c in inst["A_cost"].items()}
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_instance_json(filepath: str) -> dict:
    """Load a JSON instance and restore arc keys to (int, int) tuples."""
    with open(filepath, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["A_cost"] = {
        tuple(int(x) for x in k.split(",")): v
        for k, v in payload["A_cost"].items()
    }
    for key in ("VS", "VT", "VD", "V0", "V"):
        payload[key] = [int(x) for x in payload[key]]
    for key in ("b", "Q", "f", "theta"):
        payload[key] = {int(k): v for k, v in payload[key].items()}
    return payload


def summarize_instance(inst: dict) -> None:
    VS, VT, VD = inst["VS"], inst["VT"], inst["VD"]
    A_cost     = inst["A_cost"]
    b          = inst["b"]
    meta       = inst.get("meta", {})
    VS_set, VT_set = set(VS), set(VT)

    total_supply = sum(b[s] for s in VS)
    total_demand = sum(-b[d] for d in VD)

    n_ST = sum(1 for (i, j) in A_cost if i in VS_set and j in VT_set)
    n_SD = sum(1 for (i, j) in A_cost if i in VS_set and j not in VT_set)
    n_TD = sum(1 for (i, j) in A_cost if i in VT_set)

    print("\n--- Instance summary ---")
    print(f"VS : {VS[:4]} ...  (total {len(VS)})")
    print(f"VT : {VT[:4]} ...  (total {len(VT)})")
    print(f"VD : {VD[:4]} ...  (total {len(VD)})")
    print(f"Arcs: {len(A_cost)}  (S-T={n_ST}, S-D={n_SD}, T-D={n_TD})")
    print(f"Total supply = {total_supply:,.2f}")
    print(f"Total demand = {total_demand:,.2f}")
    print(f"Supply >= Demand? {total_supply >= total_demand}")
    if meta:
        print(f"Density (req/actual) = {meta.get('requested_density')} / "
              f"{meta.get('actual_density')}")
        print(f"s_bar = {meta.get('s_bar'):.2f},  c_bar = {meta.get('c_bar'):.2f}")
        print(f"m_budget = {meta.get('m_budget')}")


if __name__ == "__main__":
    inst = generate_instance_paper_exact(
        nS=10, nT=20, nD=30, density=0.3, seed=1, m_budget=5)
    summarize_instance(inst)
    save_instance_json(inst, "generated_instances/paper_exact_instance.json")
    print("\nSaved to generated_instances/paper_exact_instance.json")
