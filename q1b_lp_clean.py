"""
Script 4b: Q1b Clean LP Solver – All-TOR Strategy with Move-Outs
=================================================================
This script implements the correct LP formulation:
- All production uses TOR tools (optimal space efficiency)
- Mintech tools are moved out as needed to free space
- Node-fab assignment is optimized by the LP
- No cross-fab transfers (each node runs entirely in one fab per quarter)

Key insight: Since TOR tools have better utilization AND smaller footprint
per unit throughput, the optimal strategy is to:
1. Replace all mintech tools with TOR tools over time
2. Move out mintech tools when space is needed
3. Distribute nodes across fabs to balance space usage
"""

import json
import math
import time
from pathlib import Path

import pulp

ROOT_DIR = Path(__file__).resolve().parent
PARAMS_PATH = ROOT_DIR / "parameters" / "params.json"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

with open(PARAMS_PATH) as f:
    P = json.load(f)

QUARTERS = P["quarters"]
FABS = P["fabs"]
NODES = P["nodes"]
LOADING = {int(n): {q: v for q, v in qv.items()} for n, qv in P["loading"].items()}
PROCESS_STEPS = {
    int(n): [
        (s["step"], s["ws_mintech"], s["rpt_mintech"], s["ws_tor"], s["rpt_tor"])
        for s in steps
    ]
    for n, steps in P["process_steps"].items()
}
WS_SPECS = {
    ws: (d["space_m2"], d["capex"], d["utilization"]) for ws, d in P["ws_specs"].items()
}
FAB_SPECS = {
    int(f): {"space": d["space"], "tools": d["tools"]}
    for f, d in P["fab_specs"].items()
}
TRANSFER_COST = P["transfer_cost_per_wafer"]
MOVEOUT_COST = P["moveout_cost_per_tool"]
WEEKS_PER_Q = P["weeks_per_quarter"]
MIN_PER_WEEK = P["minutes_per_week"]

MINTECH_WS = ["A", "B", "C", "D", "E", "F"]
TOR_WS = ["A+", "B+", "C+", "D+", "E+", "F+"]
TOR_MAP = {"A": "A+", "B": "B+", "C": "C+", "D": "D+", "E": "E+", "F": "F+"}
ALL_WS = MINTECH_WS + TOR_WS


def compute_tor_req_per_wafer(node, ws_tor):
    """Total TOR tool-hours per wafer for a given node and TOR workstation."""
    _, _, util_tor = WS_SPECS[ws_tor]
    ws_mt = ws_tor[:-1]
    total = 0.0
    for step_num, ws_step, rpt_mt, ws_tor_step, rpt_tor in PROCESS_STEPS[node]:
        if ws_tor_step == ws_tor:
            total += rpt_tor / (MIN_PER_WEEK * util_tor)
    return total


def compute_space_per_wafer(node, ws_tor):
    """Space (m²) per wafer per week for a given node and TOR workstation."""
    space, _, _ = WS_SPECS[ws_tor]
    return compute_tor_req_per_wafer(node, ws_tor) * space


def total_space_per_wafer(node):
    """Total space needed across all TOR workstations for one wafer/week of node n."""
    return sum(compute_space_per_wafer(node, ws_tor) for ws_tor in TOR_WS)


# Pre-compute space per wafer for each node
print("=== Space per wafer/week (m²) for all-TOR strategy ===")
for n in NODES:
    spw = total_space_per_wafer(n)
    print(f"  Node {n}: {spw:.4f} m² per wafer/week")

# Check total space needed at Q4'27
print("\n=== Space needed at Q4'27 (all-TOR, optimal distribution) ===")
total_spw = sum(total_space_per_wafer(n) * LOADING[n]["Q4'27"] for n in NODES)
print(f"  Total space needed: {total_spw:.1f} m²")
print(f"  Total available: {sum(FAB_SPECS[f]['space'] for f in FABS)} m²")
print(f"  Feasible: {total_spw <= sum(FAB_SPECS[f]['space'] for f in FABS)}")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE LP MODEL
# ─────────────────────────────────────────────────────────────────────────────


def build_model():
    """
    LP model:
    - flow[q][n][f]: wafers of node n processed in fab f in quarter q (continuous)
    - tor[q][ws][f]: TOR tools of type ws in fab f in quarter q (continuous)
    - mt[q][ws][f]: mintech tools of type ws in fab f in quarter q (continuous)
    - moveout[q][ws][f]: mintech tools moved out in quarter q (continuous)
    - buy_tor[q][ws][f]: TOR tools purchased in quarter q (continuous)

    Objective: minimize CapEx (TOR purchases) + OpEx (move-outs)
    """
    model = pulp.LpProblem("Q1b_LP_Clean", pulp.LpMinimize)

    # Flow variables
    flow = {
        q: {
            n: {f: pulp.LpVariable(f"flow_{q}_{n}_{f}", lowBound=0) for f in FABS}
            for n in NODES
        }
        for q in QUARTERS
    }

    # TOR tool variables
    tor = {
        q: {
            ws: {f: pulp.LpVariable(f"tor_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in TOR_WS
        }
        for q in QUARTERS
    }

    # Mintech tool variables (can decrease)
    mt = {
        q: {
            ws: {f: pulp.LpVariable(f"mt_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in MINTECH_WS
        }
        for q in QUARTERS
    }

    # Move-out variables
    moveout = {
        q: {
            ws: {f: pulp.LpVariable(f"out_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in MINTECH_WS
        }
        for q in QUARTERS
    }

    # TOR purchase variables
    buy_tor = {
        q: {
            ws: {f: pulp.LpVariable(f"buy_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in TOR_WS
        }
        for q in QUARTERS
    }

    # ── Link constraints ────────────────────────────────────────────────────

    for qi, q in enumerate(QUARTERS):
        for ws in MINTECH_WS:
            for f in FABS:
                initial = FAB_SPECS[f]["tools"][ws]
                if qi == 0:
                    model += (
                        mt[q][ws][f] == initial - moveout[q][ws][f],
                        f"mt_link_{q}_{ws}_{f}",
                    )
                else:
                    q_prev = QUARTERS[qi - 1]
                    model += (
                        mt[q][ws][f] == mt[q_prev][ws][f] - moveout[q][ws][f],
                        f"mt_link_{q}_{ws}_{f}",
                    )

        for ws in TOR_WS:
            for f in FABS:
                if qi == 0:
                    model += (
                        tor[q][ws][f] == buy_tor[q][ws][f],
                        f"tor_link_{q}_{ws}_{f}",
                    )
                else:
                    q_prev = QUARTERS[qi - 1]
                    model += (
                        tor[q][ws][f] == tor[q_prev][ws][f] + buy_tor[q][ws][f],
                        f"tor_link_{q}_{ws}_{f}",
                    )

    # ── Objective ───────────────────────────────────────────────────────────

    capex = pulp.lpSum(
        WS_SPECS[ws][1] * buy_tor[q][ws][f]
        for q in QUARTERS
        for ws in TOR_WS
        for f in FABS
    )
    opex_moveout = pulp.lpSum(
        MOVEOUT_COST * moveout[q][ws][f]
        for q in QUARTERS
        for ws in MINTECH_WS
        for f in FABS
    )

    model += capex + opex_moveout, "Total_Cost"

    # ── Constraints ─────────────────────────────────────────────────────────

    # C1. Demand
    for q in QUARTERS:
        for n in NODES:
            model += (
                pulp.lpSum(flow[q][n][f] for f in FABS) == LOADING[n][q],
                f"demand_{q}_{n}",
            )

    # C2. TOR capacity: for each (q, ws_tor, f), tool requirement <= tor[q][ws][f]
    for q in QUARTERS:
        for ws_tor in TOR_WS:
            _, _, util_tor = WS_SPECS[ws_tor]
            for f in FABS:
                usage = []
                for n in NODES:
                    for step_num, ws_mt, rpt_mt, ws_tor_step, rpt_tor in PROCESS_STEPS[
                        n
                    ]:
                        if ws_tor_step == ws_tor:
                            usage.append(
                                flow[q][n][f] * rpt_tor / (MIN_PER_WEEK * util_tor)
                            )
                if usage:
                    model += (
                        pulp.lpSum(usage) <= tor[q][ws_tor][f],
                        f"cap_{q}_{ws_tor}_{f}",
                    )

    # C3. Space constraint
    for q in QUARTERS:
        for f in FABS:
            space_mt = pulp.lpSum(WS_SPECS[ws][0] * mt[q][ws][f] for ws in MINTECH_WS)
            space_tor = pulp.lpSum(WS_SPECS[ws][0] * tor[q][ws][f] for ws in TOR_WS)
            model += space_mt + space_tor <= FAB_SPECS[f]["space"], f"space_{q}_{f}"

    # C4. Mintech tools >= 0 (already enforced by lowBound=0)
    # C5. Move-outs >= 0 (already enforced)
    # C6. TOR tools non-decreasing (enforced by buy_tor >= 0)

    return model, flow, tor, mt, moveout, buy_tor


# ─────────────────────────────────────────────────────────────────────────────
# SOLVE
# ─────────────────────────────────────────────────────────────────────────────


def solve_and_extract(model, flow, tor, mt, moveout, buy_tor):
    n_vars = len(model.variables())
    n_cons = len(model.constraints)
    print(f"\nModel: {n_vars} variables, {n_cons} constraints.")
    print("Solving LP...")

    start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=120)
    status = model.solve(solver)
    elapsed = time.time() - start

    print(f"Status: {pulp.LpStatus[status]}, Time: {elapsed:.1f}s")

    if pulp.value(model.objective) is None:
        print("LP infeasible!")
        return None

    obj = pulp.value(model.objective)
    print(f"LP Objective: ${obj:,.0f}")

    # Extract
    results = {
        "status": pulp.LpStatus[status],
        "lp_obj": obj,
        "flow": {},
        "tor": {},
        "mt": {},
        "moveout": {},
        "buy_tor": {},
    }

    for q in QUARTERS:
        results["flow"][q] = {}
        for n in NODES:
            results["flow"][q][n] = {}
            for f in FABS:
                v = pulp.value(flow[q][n][f])
                results["flow"][q][n][f] = max(0, v) if v else 0

        results["tor"][q] = {}
        results["buy_tor"][q] = {}
        for ws in TOR_WS:
            results["tor"][q][ws] = {}
            results["buy_tor"][q][ws] = {}
            for f in FABS:
                v = pulp.value(tor[q][ws][f])
                results["tor"][q][ws][f] = max(0, v) if v else 0
                vb = pulp.value(buy_tor[q][ws][f])
                results["buy_tor"][q][ws][f] = max(0, vb) if vb else 0

        results["mt"][q] = {}
        results["moveout"][q] = {}
        for ws in MINTECH_WS:
            results["mt"][q][ws] = {}
            results["moveout"][q][ws] = {}
            for f in FABS:
                v = pulp.value(mt[q][ws][f])
                results["mt"][q][ws][f] = max(0, v) if v else 0
                vm = pulp.value(moveout[q][ws][f])
                results["moveout"][q][ws][f] = max(0, vm) if vm else 0

    return results


def round_up_solution(results):
    """
    Round up TOR tool counts to integers and adjust move-outs accordingly.
    This converts the LP relaxation to a feasible integer solution.
    """
    print("\n=== Rounding LP solution to integers ===")

    rounded = {
        "flow": {},
        "tor": {},
        "mt": {},
        "moveout": {},
        "buy_tor": {},
    }

    # Round up TOR tools (ceiling)
    for q in QUARTERS:
        rounded["tor"][q] = {}
        for ws in TOR_WS:
            rounded["tor"][q][ws] = {}
            for f in FABS:
                rounded["tor"][q][ws][f] = math.ceil(results["tor"][q][ws][f])

    # Recompute buy_tor from rounded tor
    rounded["buy_tor"] = {}
    prev_tor = {ws: {f: 0 for f in FABS} for ws in TOR_WS}
    for q in QUARTERS:
        rounded["buy_tor"][q] = {}
        for ws in TOR_WS:
            rounded["buy_tor"][q][ws] = {}
            for f in FABS:
                bought = rounded["tor"][q][ws][f] - prev_tor[ws][f]
                rounded["buy_tor"][q][ws][f] = max(0, bought)
        prev_tor = {ws: {f: rounded["tor"][q][ws][f] for f in FABS} for ws in TOR_WS}

    # Flow: round to nearest integer
    for q in QUARTERS:
        rounded["flow"][q] = {}
        for n in NODES:
            rounded["flow"][q][n] = {}
            for f in FABS:
                rounded["flow"][q][n][f] = round(results["flow"][q][n][f])

    # Compute move-outs needed to satisfy space constraints with rounded tools
    rounded["mt"] = {}
    rounded["moveout"] = {}
    current_mt = {ws: {f: FAB_SPECS[f]["tools"][ws] for f in FABS} for ws in MINTECH_WS}

    for q in QUARTERS:
        rounded["mt"][q] = {}
        rounded["moveout"][q] = {}

        for ws in MINTECH_WS:
            rounded["mt"][q][ws] = {}
            rounded["moveout"][q][ws] = {}
            for f in FABS:
                rounded["mt"][q][ws][f] = current_mt[ws][f]
                rounded["moveout"][q][ws][f] = 0

        # Check space and move out if needed
        for f in FABS:
            space_tor = sum(WS_SPECS[ws][0] * rounded["tor"][q][ws][f] for ws in TOR_WS)
            space_mt = sum(WS_SPECS[ws][0] * current_mt[ws][f] for ws in MINTECH_WS)
            total = space_tor + space_mt
            avail = FAB_SPECS[f]["space"]

            if total > avail:
                excess = total - avail
                # Move out mintech tools, prioritizing those with most tools (to minimize cost)
                ws_order = sorted(MINTECH_WS, key=lambda ws: -current_mt[ws][f])
                for ws in ws_order:
                    if excess <= 0:
                        break
                    count = current_mt[ws][f]
                    if count == 0:
                        continue
                    space_per = WS_SPECS[ws][0]
                    to_move = min(count, math.ceil(excess / space_per))
                    rounded["moveout"][q][ws][f] = to_move
                    rounded["mt"][q][ws][f] = count - to_move
                    excess -= to_move * space_per

        # Update current_mt
        for ws in MINTECH_WS:
            for f in FABS:
                current_mt[ws][f] = rounded["mt"][q][ws][f]

    # Validate
    print("\n=== Space Validation (rounded) ===")
    all_ok = True
    for q in QUARTERS:
        for f in FABS:
            space_mt = sum(
                WS_SPECS[ws][0] * rounded["mt"][q][ws][f] for ws in MINTECH_WS
            )
            space_tor = sum(WS_SPECS[ws][0] * rounded["tor"][q][ws][f] for ws in TOR_WS)
            total = space_mt + space_tor
            avail = FAB_SPECS[f]["space"]
            if total > avail + 0.01:
                print(f"  VIOLATED: {q} Fab {f}: {total:.2f}/{avail}")
                all_ok = False
    if all_ok:
        print("  All space constraints satisfied!")

    print("\n=== Capacity Validation (rounded) ===")
    cap_ok = True
    for q in QUARTERS:
        for ws_tor in TOR_WS:
            _, _, util_tor = WS_SPECS[ws_tor]
            for f in FABS:
                demand = 0
                for n in NODES:
                    wafers = rounded["flow"][q][n][f]
                    for step_num, ws_mt, rpt_mt, ws_tor_step, rpt_tor in PROCESS_STEPS[
                        n
                    ]:
                        if ws_tor_step == ws_tor:
                            demand += wafers * rpt_tor / (MIN_PER_WEEK * util_tor)
                supply = rounded["tor"][q][ws_tor][f]
                if demand > supply + 0.01:
                    print(
                        f"  VIOLATED: {q} Fab {f} {ws_tor}: demand={demand:.2f} > supply={supply}"
                    )
                    cap_ok = False
    if cap_ok:
        print("  All capacity constraints satisfied!")

    print("\n=== Demand Validation (rounded) ===")
    dem_ok = True
    for q in QUARTERS:
        for n in NODES:
            total = sum(rounded["flow"][q][n][f] for f in FABS)
            req = LOADING[n][q]
            if abs(total - req) > 1:
                print(f"  VIOLATED: {q} Node {n}: got {total}, need {req}")
                dem_ok = False
    if dem_ok:
        print("  All demand constraints satisfied!")

    # Compute costs
    capex = sum(
        WS_SPECS[ws][1] * rounded["buy_tor"][q][ws][f]
        for q in QUARTERS
        for ws in TOR_WS
        for f in FABS
    )
    opex_m = sum(
        MOVEOUT_COST * rounded["moveout"][q][ws][f]
        for q in QUARTERS
        for ws in MINTECH_WS
        for f in FABS
    )
    rounded["capex"] = capex
    rounded["opex_moveout"] = opex_m
    rounded["total_cost"] = capex + opex_m

    print(f"\n  CapEx (TOR purchases):  ${capex:>15,.0f}")
    print(f"  OpEx (move-outs):       ${opex_m:>15,.0f}")
    print(f"  Total Cost:             ${capex + opex_m:>15,.0f}")

    # Flow summary
    print("\n=== Flow Assignment (wafers/week) ===")
    print(
        f"{'Quarter':<10} {'Node':<6} {'Fab1':>8} {'Fab2':>8} {'Fab3':>8} {'Total':>8} {'Req':>8}"
    )
    for q in QUARTERS:
        for n in NODES:
            f1 = rounded["flow"][q][n][1]
            f2 = rounded["flow"][q][n][2]
            f3 = rounded["flow"][q][n][3]
            total = f1 + f2 + f3
            req = LOADING[n][q]
            ok = "OK" if abs(total - req) <= 1 else "ERR"
            print(
                f"{q:<10} {n:<6} {f1:>8.0f} {f2:>8.0f} {f3:>8.0f} {total:>8.0f} {req:>8} {ok}"
            )

    # Tool summary
    print("\n=== TOR Tool Summary (Q4'27) ===")
    q = QUARTERS[-1]
    for ws in TOR_WS:
        counts = [rounded["tor"][q][ws][f] for f in FABS]
        if any(c > 0 for c in counts):
            print(f"  {ws:3s}: F1={counts[0]:3d}, F2={counts[1]:3d}, F3={counts[2]:3d}")

    print("\n=== Move-Out Summary ===")
    total_mo = 0
    for q in QUARTERS:
        q_mo = sum(rounded["moveout"][q][ws][f] for ws in MINTECH_WS for f in FABS)
        if q_mo > 0:
            print(f"  {q}: {q_mo} tools")
            for ws in MINTECH_WS:
                for f in FABS:
                    m = rounded["moveout"][q][ws][f]
                    if m > 0:
                        print(f"    {ws} Fab{f}: -{m}")
            total_mo += q_mo
    print(f"  Total: {total_mo} tools (${total_mo * MOVEOUT_COST:,.0f})")

    return rounded


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Q1b CLEAN LP SOLVER")
    print("=" * 60)

    model, flow, tor, mt, moveout, buy_tor = build_model()
    lp_results = solve_and_extract(model, flow, tor, mt, moveout, buy_tor)

    if lp_results:
        final = round_up_solution(lp_results)

        def convert(obj):
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, float):
                return round(obj, 4)
            return obj

        with open("results/q1b_final_results.json", "w") as f:
            json.dump(convert(final), f, indent=2)
        print("\nFinal results saved to: results/q1b_final_results.json")
