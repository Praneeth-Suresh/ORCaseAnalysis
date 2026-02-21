"""
Script 3: Q1b MILP Solver – Move-Outs Allowed, Both Mintech and TOR
====================================================================
Solves the production and tool allocation problem for Q1b:
  - Tool move-outs allowed ($1M per tool)
  - Both mintech and TOR tools can be purchased
  - Minimizes CapEx (new tool purchases) + OpEx (cross-fab transfers + move-outs)
  - All demand must be met; fab space constraints must be respected

Key insight from Q1a analysis: The problem is infeasible without move-outs
because keeping all initial mintech tools (2,869 m²) plus the required new
TOR tools (3,293 m² at Q4'27) would need 6,162 m² vs 3,500 m² available.
Move-outs are therefore NECESSARY for feasibility.

Model design:
  - total_mt[q][ws][f]: total mintech tools of type ws in fab f at quarter q
  - total_tor[q][ws][f]: total TOR tools of type ws in fab f at quarter q
  - flow_mt[q][n][s][f]: wafers processed on mintech at step s, fab f, quarter q
  - flow_tor[q][n][s][f]: wafers processed on TOR at step s, fab f, quarter q
  - Tools can be moved out (decreasing total_mt) but cannot be added back
  - New tools can be purchased (increasing total_mt or total_tor)
"""

import json
import math
import time
from pathlib import Path

import pulp

ROOT_DIR = Path(__file__).resolve().parent
PARAMS_PATH = ROOT_DIR / "parameters" / "params.json"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PARAMETERS
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

# Upper bounds for tool counts (to help solver)
# Max tools that could fit in a single fab
MAX_TOOLS = {ws: int(1500 / WS_SPECS[ws][0]) + 10 for ws in ALL_WS}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE MILP MODEL
# ─────────────────────────────────────────────────────────────────────────────


def build_q1b_model():
    print("Building Q1b MILP model (mintech + TOR, move-outs allowed)...")
    model = pulp.LpProblem("Q1b_Production_Optimization", pulp.LpMinimize)

    # ── Decision Variables ──────────────────────────────────────────────────

    # Flow variables: wafers processed on mintech vs TOR at each step/fab/quarter
    flow_mt = {}
    flow_tor = {}
    for q in QUARTERS:
        flow_mt[q] = {}
        flow_tor[q] = {}
        for n in NODES:
            flow_mt[q][n] = {}
            flow_tor[q][n] = {}
            for step_num, ws, rpt, ws_tor, rpt_tor in PROCESS_STEPS[n]:
                flow_mt[q][n][step_num] = {
                    f: pulp.LpVariable(f"fmt_{q}_{n}_{step_num}_{f}", lowBound=0)
                    for f in FABS
                }
                flow_tor[q][n][step_num] = {
                    f: pulp.LpVariable(f"ftor_{q}_{n}_{step_num}_{f}", lowBound=0)
                    for f in FABS
                }

    # Tool count variables (total tools in fab at start of each quarter)
    total_mt = {}  # mintech tools
    total_tor = {}  # TOR tools
    for q in QUARTERS:
        total_mt[q] = {}
        total_tor[q] = {}
        for ws in MINTECH_WS:
            total_mt[q][ws] = {
                f: pulp.LpVariable(f"mt_{q}_{ws}_{f}", lowBound=0, cat="Integer")
                for f in FABS
            }
        for ws in TOR_WS:
            total_tor[q][ws] = {
                f: pulp.LpVariable(f"tor_{q}_{ws}_{f}", lowBound=0, cat="Integer")
                for f in FABS
            }

    # Transfer variables
    transfer = {}
    for q in QUARTERS:
        transfer[q] = {}
        for n in NODES:
            transfer[q][n] = {}
            steps = PROCESS_STEPS[n]
            for idx in range(1, len(steps)):
                s_prev = steps[idx - 1][0]
                s_curr = steps[idx][0]
                key = (s_prev, s_curr)
                transfer[q][n][key] = {
                    f_from: {
                        f_to: pulp.LpVariable(
                            f"xfer_{q}_{n}_{s_prev}_{s_curr}_{f_from}_{f_to}",
                            lowBound=0,
                        )
                        for f_to in FABS
                    }
                    for f_from in FABS
                }

    # ── Objective Function ──────────────────────────────────────────────────

    # CapEx: new tool purchases (incremental)
    capex_terms = []
    for qi, q in enumerate(QUARTERS):
        for ws in MINTECH_WS:
            _, capex, _ = WS_SPECS[ws]
            for f in FABS:
                initial = FAB_SPECS[f]["tools"][ws]
                if qi == 0:
                    # New purchases in Q1 = total_mt[Q1] - initial (if positive)
                    # Move-outs in Q1 = initial - total_mt[Q1] (if positive)
                    # We handle this via the move-out cost below
                    pass
                # Purchases = max(0, total_mt[q] - total_mt[q_prev])
                # We use auxiliary variable for this
        for ws in TOR_WS:
            _, capex, _ = WS_SPECS[ws]
            for f in FABS:
                if qi == 0:
                    bought = total_tor[q][ws][f]  # all TOR tools in Q1 are new
                else:
                    q_prev = QUARTERS[qi - 1]
                    bought = total_tor[q][ws][f] - total_tor[q_prev][ws][f]
                capex_terms.append(capex * bought)

    # For mintech purchases, we need to track incremental purchases separately
    # from move-outs. We introduce purchase variables.
    purchase_mt = {}
    moveout_mt = {}
    for qi, q in enumerate(QUARTERS):
        purchase_mt[q] = {}
        moveout_mt[q] = {}
        for ws in MINTECH_WS:
            purchase_mt[q][ws] = {}
            moveout_mt[q][ws] = {}
            for f in FABS:
                purchase_mt[q][ws][f] = pulp.LpVariable(
                    f"buy_mt_{q}_{ws}_{f}", lowBound=0, cat="Integer"
                )
                moveout_mt[q][ws][f] = pulp.LpVariable(
                    f"out_mt_{q}_{ws}_{f}", lowBound=0, cat="Integer"
                )

    # Link purchase/moveout to total_mt
    for qi, q in enumerate(QUARTERS):
        for ws in MINTECH_WS:
            for f in FABS:
                initial = FAB_SPECS[f]["tools"][ws]
                if qi == 0:
                    # total_mt[Q1] = initial + purchase - moveout
                    model += (
                        total_mt[q][ws][f]
                        == initial + purchase_mt[q][ws][f] - moveout_mt[q][ws][f],
                        f"mt_link_{q}_{ws}_{f}",
                    )
                else:
                    q_prev = QUARTERS[qi - 1]
                    model += (
                        total_mt[q][ws][f]
                        == total_mt[q_prev][ws][f]
                        + purchase_mt[q][ws][f]
                        - moveout_mt[q][ws][f],
                        f"mt_link_{q}_{ws}_{f}",
                    )

    # TOR tools can only be purchased (not moved out in this model)
    # Link TOR purchases
    purchase_tor = {}
    for qi, q in enumerate(QUARTERS):
        purchase_tor[q] = {}
        for ws in TOR_WS:
            purchase_tor[q][ws] = {}
            for f in FABS:
                purchase_tor[q][ws][f] = pulp.LpVariable(
                    f"buy_tor_{q}_{ws}_{f}", lowBound=0, cat="Integer"
                )
                if qi == 0:
                    model += (
                        total_tor[q][ws][f] == purchase_tor[q][ws][f],
                        f"tor_link_{q}_{ws}_{f}",
                    )
                else:
                    q_prev = QUARTERS[qi - 1]
                    model += (
                        total_tor[q][ws][f]
                        == total_tor[q_prev][ws][f] + purchase_tor[q][ws][f],
                        f"tor_link_{q}_{ws}_{f}",
                    )

    # CapEx from mintech purchases
    for q in QUARTERS:
        for ws in MINTECH_WS:
            _, capex, _ = WS_SPECS[ws]
            for f in FABS:
                capex_terms.append(capex * purchase_mt[q][ws][f])

    # OpEx: transfer costs
    opex_transfer_terms = []
    for q in QUARTERS:
        for n in NODES:
            steps = PROCESS_STEPS[n]
            for idx in range(1, len(steps)):
                s_prev = steps[idx - 1][0]
                s_curr = steps[idx][0]
                key = (s_prev, s_curr)
                for f_from in FABS:
                    for f_to in FABS:
                        if f_from != f_to:
                            t = transfer[q][n][key][f_from][f_to]
                            opex_transfer_terms.append(TRANSFER_COST * WEEKS_PER_Q * t)

    # OpEx: move-out costs
    opex_moveout_terms = []
    for q in QUARTERS:
        for ws in MINTECH_WS:
            for f in FABS:
                opex_moveout_terms.append(MOVEOUT_COST * moveout_mt[q][ws][f])

    model += (
        pulp.lpSum(capex_terms)
        + pulp.lpSum(opex_transfer_terms)
        + pulp.lpSum(opex_moveout_terms),
        "Total_Cost",
    )

    # ── Constraints ─────────────────────────────────────────────────────────

    def total_flow(q, n, s, f):
        return flow_mt[q][n][s][f] + flow_tor[q][n][s][f]

    # C1. Demand
    for q in QUARTERS:
        for n in NODES:
            first_step = PROCESS_STEPS[n][0][0]
            model += (
                pulp.lpSum(total_flow(q, n, first_step, f) for f in FABS)
                == LOADING[n][q],
                f"demand_{q}_{n}",
            )

    # C2. Flow conservation (total wafers at each step = loading)
    for q in QUARTERS:
        for n in NODES:
            loading_n = LOADING[n][q]
            for step_num, *_ in PROCESS_STEPS[n]:
                model += (
                    pulp.lpSum(total_flow(q, n, step_num, f) for f in FABS)
                    == loading_n,
                    f"flow_total_{q}_{n}_{step_num}",
                )

    # C3. Transfer balance
    for q in QUARTERS:
        for n in NODES:
            steps = PROCESS_STEPS[n]
            for idx in range(1, len(steps)):
                s_prev = steps[idx - 1][0]
                s_curr = steps[idx][0]
                key = (s_prev, s_curr)
                for f in FABS:
                    outgoing = pulp.lpSum(
                        transfer[q][n][key][f][f_to] for f_to in FABS if f_to != f
                    )
                    incoming = pulp.lpSum(
                        transfer[q][n][key][f_from][f] for f_from in FABS if f_from != f
                    )
                    model += (
                        total_flow(q, n, s_curr, f)
                        == total_flow(q, n, s_prev, f) - outgoing + incoming,
                        f"xfer_balance_{q}_{n}_{s_prev}_{s_curr}_{f}",
                    )

    # C4. Mintech tool capacity
    for q in QUARTERS:
        for ws in MINTECH_WS:
            _, _, util = WS_SPECS[ws]
            for f in FABS:
                usage = []
                for n in NODES:
                    for step_num, ws_step, rpt, ws_tor, rpt_tor in PROCESS_STEPS[n]:
                        if ws_step == ws:
                            usage.append(
                                flow_mt[q][n][step_num][f] * rpt / (MIN_PER_WEEK * util)
                            )
                if usage:
                    model += (
                        pulp.lpSum(usage) <= total_mt[q][ws][f],
                        f"cap_mt_{q}_{ws}_{f}",
                    )

    # C5. TOR tool capacity
    for q in QUARTERS:
        for ws in MINTECH_WS:
            ws_tor = TOR_MAP[ws]
            _, _, util_tor = WS_SPECS[ws_tor]
            for f in FABS:
                usage = []
                for n in NODES:
                    for step_num, ws_step, rpt, ws_tor_step, rpt_tor in PROCESS_STEPS[
                        n
                    ]:
                        if ws_step == ws:
                            usage.append(
                                flow_tor[q][n][step_num][f]
                                * rpt_tor
                                / (MIN_PER_WEEK * util_tor)
                            )
                if usage:
                    model += (
                        pulp.lpSum(usage) <= total_tor[q][ws_tor][f],
                        f"cap_tor_{q}_{ws_tor}_{f}",
                    )

    # C6. Fab space constraint
    for q in QUARTERS:
        for f in FABS:
            space_terms = []
            for ws in MINTECH_WS:
                space, _, _ = WS_SPECS[ws]
                space_terms.append(space * total_mt[q][ws][f])
            for ws in TOR_WS:
                space, _, _ = WS_SPECS[ws]
                space_terms.append(space * total_tor[q][ws][f])
            model += (
                pulp.lpSum(space_terms) <= FAB_SPECS[f]["space"],
                f"space_{q}_{f}",
            )

    # C7. Once a tool is moved out, it cannot be brought back
    # (enforced by the move-out being permanent: total_mt can decrease but purchase_mt >= 0)
    # This is already handled by the link constraints above.

    return (
        model,
        flow_mt,
        flow_tor,
        total_mt,
        total_tor,
        transfer,
        purchase_mt,
        moveout_mt,
        purchase_tor,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SOLVE AND EXTRACT RESULTS
# ─────────────────────────────────────────────────────────────────────────────


def solve_and_report(
    model,
    flow_mt,
    flow_tor,
    total_mt,
    total_tor,
    transfer,
    purchase_mt,
    moveout_mt,
    purchase_tor,
):
    n_vars = len(model.variables())
    n_cons = len(model.constraints)
    print(f"\nModel: {n_vars} variables, {n_cons} constraints.")
    print("Solving with CBC (time limit: 600s, gap: 2%)...")

    start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=600, gapRel=0.02)
    status = model.solve(solver)
    elapsed = time.time() - start

    print(f"\nSolver status: {pulp.LpStatus[status]}")
    print(f"Solve time: {elapsed:.1f}s")

    if pulp.value(model.objective) is None:
        print("No feasible solution found.")
        return None

    obj_val = pulp.value(model.objective)
    print(f"Objective (Total Cost): ${obj_val:,.0f}")

    # Extract results
    results = {
        "status": pulp.LpStatus[status],
        "total_cost": obj_val,
        "flow_mt": {},
        "flow_tor": {},
        "total_mt": {},
        "total_tor": {},
        "purchase_mt": {},
        "moveout_mt": {},
        "purchase_tor": {},
        "capex": 0,
        "opex_transfer": 0,
        "opex_moveout": 0,
    }

    for q in QUARTERS:
        results["flow_mt"][q] = {}
        results["flow_tor"][q] = {}
        for n in NODES:
            results["flow_mt"][q][n] = {}
            results["flow_tor"][q][n] = {}
            for step_num, *_ in PROCESS_STEPS[n]:
                results["flow_mt"][q][n][step_num] = {}
                results["flow_tor"][q][n][step_num] = {}
                for f in FABS:
                    v_mt = pulp.value(flow_mt[q][n][step_num][f])
                    v_tor = pulp.value(flow_tor[q][n][step_num][f])
                    results["flow_mt"][q][n][step_num][f] = (
                        max(0, round(v_mt)) if v_mt else 0
                    )
                    results["flow_tor"][q][n][step_num][f] = (
                        max(0, round(v_tor)) if v_tor else 0
                    )

    for q in QUARTERS:
        results["total_mt"][q] = {}
        results["total_tor"][q] = {}
        results["purchase_mt"][q] = {}
        results["moveout_mt"][q] = {}
        results["purchase_tor"][q] = {}
        for ws in MINTECH_WS:
            results["total_mt"][q][ws] = {}
            results["purchase_mt"][q][ws] = {}
            results["moveout_mt"][q][ws] = {}
            for f in FABS:
                v = pulp.value(total_mt[q][ws][f])
                results["total_mt"][q][ws][f] = max(0, round(v)) if v else 0
                vp = pulp.value(purchase_mt[q][ws][f])
                results["purchase_mt"][q][ws][f] = max(0, round(vp)) if vp else 0
                vm = pulp.value(moveout_mt[q][ws][f])
                results["moveout_mt"][q][ws][f] = max(0, round(vm)) if vm else 0
        for ws in TOR_WS:
            results["total_tor"][q][ws] = {}
            results["purchase_tor"][q][ws] = {}
            for f in FABS:
                v = pulp.value(total_tor[q][ws][f])
                results["total_tor"][q][ws][f] = max(0, round(v)) if v else 0
                vp = pulp.value(purchase_tor[q][ws][f])
                results["purchase_tor"][q][ws][f] = max(0, round(vp)) if vp else 0

    # Compute costs
    capex = 0
    for q in QUARTERS:
        for ws in MINTECH_WS:
            _, cost, _ = WS_SPECS[ws]
            for f in FABS:
                capex += results["purchase_mt"][q][ws][f] * cost
        for ws in TOR_WS:
            _, cost, _ = WS_SPECS[ws]
            for f in FABS:
                capex += results["purchase_tor"][q][ws][f] * cost
    results["capex"] = capex

    opex_t = 0
    for q in QUARTERS:
        for n in NODES:
            steps = PROCESS_STEPS[n]
            for idx in range(1, len(steps)):
                s_prev = steps[idx - 1][0]
                s_curr = steps[idx][0]
                key = (s_prev, s_curr)
                for f_from in FABS:
                    for f_to in FABS:
                        if f_from != f_to:
                            val = pulp.value(transfer[q][n][key][f_from][f_to])
                            if val and val > 0.5:
                                opex_t += val * TRANSFER_COST * WEEKS_PER_Q
    results["opex_transfer"] = opex_t

    opex_m = 0
    for q in QUARTERS:
        for ws in MINTECH_WS:
            for f in FABS:
                opex_m += results["moveout_mt"][q][ws][f] * MOVEOUT_COST
    results["opex_moveout"] = opex_m

    print(f"\n  CapEx (new tools):      ${capex:>15,.0f}")
    print(f"  OpEx (transfers):       ${opex_t:>15,.0f}")
    print(f"  OpEx (move-outs):       ${opex_m:>15,.0f}")
    print(f"  Total Cost:             ${capex + opex_t + opex_m:>15,.0f}")

    # Validate space
    print("\n=== Space Validation ===")
    all_ok = True
    for q in QUARTERS:
        for f in FABS:
            used = sum(
                WS_SPECS[ws][0] * results["total_mt"][q][ws][f] for ws in MINTECH_WS
            )
            used += sum(
                WS_SPECS[ws][0] * results["total_tor"][q][ws][f] for ws in TOR_WS
            )
            avail = FAB_SPECS[f]["space"]
            if used > avail + 0.1:
                print(f"  VIOLATED: {q} Fab {f}: {used:.1f}/{avail} m²")
                all_ok = False
    if all_ok:
        print("  All space constraints satisfied!")

    # Validate demand
    print("\n=== Demand Validation ===")
    demand_ok = True
    for q in QUARTERS:
        for n in NODES:
            first_step = PROCESS_STEPS[n][0][0]
            total = sum(
                results["flow_mt"][q][n][first_step][f]
                + results["flow_tor"][q][n][first_step][f]
                for f in FABS
            )
            required = LOADING[n][q]
            if abs(total - required) > 1:
                print(f"  VIOLATED: {q} Node {n}: {total} vs {required}")
                demand_ok = False
    if demand_ok:
        print("  All demand constraints satisfied!")

    # Print tool summary
    print("\n=== Tool Summary (Q1'26 and Q4'27) ===")
    for q in [QUARTERS[0], QUARTERS[-1]]:
        print(f"\n  {q}:")
        for ws in MINTECH_WS:
            mt = [results["total_mt"][q][ws][f] for f in FABS]
            tor_ws = TOR_MAP[ws]
            tor = [results["total_tor"][q][tor_ws][f] for f in FABS]
            init = [FAB_SPECS[f]["tools"][ws] for f in FABS]
            if any(mt[i] > 0 or tor[i] > 0 for i in range(3)):
                print(
                    f"    {ws:3s}: MT=[{mt[0]:3d},{mt[1]:3d},{mt[2]:3d}] "
                    f"TOR=[{tor[0]:3d},{tor[1]:3d},{tor[2]:3d}]"
                )

    # Print move-out summary
    print("\n=== Move-Out Summary ===")
    total_moveouts = 0
    for q in QUARTERS:
        q_moveouts = sum(
            results["moveout_mt"][q][ws][f] for ws in MINTECH_WS for f in FABS
        )
        if q_moveouts > 0:
            print(f"  {q}: {q_moveouts} tools moved out")
            for ws in MINTECH_WS:
                for f in FABS:
                    m = results["moveout_mt"][q][ws][f]
                    if m > 0:
                        print(f"    {ws} Fab{f}: -{m}")
            total_moveouts += q_moveouts
    print(
        f"  Total move-outs: {total_moveouts} tools (${total_moveouts * MOVEOUT_COST:,.0f})"
    )

    return results


def save_results(results, path):
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        return obj

    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    (
        model,
        flow_mt,
        flow_tor,
        total_mt,
        total_tor,
        transfer,
        purchase_mt,
        moveout_mt,
        purchase_tor,
    ) = build_q1b_model()
    results = solve_and_report(
        model,
        flow_mt,
        flow_tor,
        total_mt,
        total_tor,
        transfer,
        purchase_mt,
        moveout_mt,
        purchase_tor,
    )
    if results:
        save_results(results, "results/q1b_results.json")
        print("\nQ1b optimization complete.")
