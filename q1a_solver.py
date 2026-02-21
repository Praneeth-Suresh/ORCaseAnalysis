"""
Script 2: Q1a MILP Solver – No Move-Outs, Both Mintech and TOR Allowed
=======================================================================
Solves the production and tool allocation problem for Q1a:
  - No tool move-outs allowed (existing tools stay)
  - Both mintech and TOR tools can be purchased
  - Minimizes CapEx (new tool purchases) + OpEx (cross-fab transfers)
  - All demand must be met; fab space constraints must be respected

Key insight: The problem is infeasible with mintech-only tools due to space
constraints in later quarters (Node 3 demand grows to 16,000/week by Q4'27,
requiring massive C workstation capacity). TOR tools are more space-efficient
per unit of throughput, making them essential.

Model design:
  - For each workstation, we track TOTAL tools (mintech + TOR) separately
  - A wafer step can be processed on either mintech or TOR tools
  - We introduce split flow variables: flow_mt[q][n][s][f] and flow_tor[q][n][s][f]
  - Tool capacity is checked separately for mintech and TOR
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
WEEKS_PER_Q = P["weeks_per_quarter"]
MIN_PER_WEEK = P["minutes_per_week"]

MINTECH_WS = ["A", "B", "C", "D", "E", "F"]
TOR_WS = ["A+", "B+", "C+", "D+", "E+", "F+"]
TOR_MAP = {"A": "A+", "B": "B+", "C": "C+", "D": "D+", "E": "E+", "F": "F+"}
ALL_WS = MINTECH_WS + TOR_WS


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE MILP MODEL
# ─────────────────────────────────────────────────────────────────────────────


def build_q1a_model():
    print("Building Q1a MILP model (mintech + TOR, no move-outs)...")
    model = pulp.LpProblem("Q1a_Production_Optimization", pulp.LpMinimize)

    # ── Decision Variables ──────────────────────────────────────────────────

    # flow_mt[q][n][s][f] – wafers processed on MINTECH tools at step s, fab f, quarter q
    # flow_tor[q][n][s][f] – wafers processed on TOR tools at step s, fab f, quarter q
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

    # total_tools[q][ws][f] – total tools of type ws in fab f in quarter q (integer)
    total_tools = {}
    for q in QUARTERS:
        total_tools[q] = {}
        for ws in ALL_WS:
            total_tools[q][ws] = {
                f: pulp.LpVariable(f"tools_{q}_{ws}_{f}", lowBound=0, cat="Integer")
                for f in FABS
            }

    # transfer variables for cross-fab moves between steps
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

    capex_terms = []
    for qi, q in enumerate(QUARTERS):
        for ws in ALL_WS:
            _, capex, _ = WS_SPECS[ws]
            for f in FABS:
                initial = FAB_SPECS[f]["tools"][ws]
                if qi == 0:
                    bought = total_tools[q][ws][f] - initial
                else:
                    q_prev = QUARTERS[qi - 1]
                    bought = total_tools[q][ws][f] - total_tools[q_prev][ws][f]
                capex_terms.append(capex * bought)

    opex_terms = []
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
                            opex_terms.append(TRANSFER_COST * WEEKS_PER_Q * t)

    model += pulp.lpSum(capex_terms) + pulp.lpSum(opex_terms), "Total_Cost"

    # ── Constraints ─────────────────────────────────────────────────────────

    # Helper: total flow at a step in a fab (mt + tor)
    def total_flow(q, n, s, f):
        return flow_mt[q][n][s][f] + flow_tor[q][n][s][f]

    # C1. Demand: total wafers at step 1 = loading
    for q in QUARTERS:
        for n in NODES:
            first_step = PROCESS_STEPS[n][0][0]
            model += (
                pulp.lpSum(total_flow(q, n, first_step, f) for f in FABS)
                == LOADING[n][q],
                f"demand_{q}_{n}",
            )

    # C2. Flow conservation: total wafers at each step = loading
    for q in QUARTERS:
        for n in NODES:
            loading_n = LOADING[n][q]
            for step_num, *_ in PROCESS_STEPS[n]:
                model += (
                    pulp.lpSum(total_flow(q, n, step_num, f) for f in FABS)
                    == loading_n,
                    f"flow_total_{q}_{n}_{step_num}",
                )

    # C3. Transfer balance: flow at step s_curr[f] = flow at s_prev[f] - out + in
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
                usage_terms = []
                for n in NODES:
                    for step_num, ws_step, rpt, ws_tor, rpt_tor in PROCESS_STEPS[n]:
                        if ws_step == ws:
                            usage_terms.append(
                                flow_mt[q][n][step_num][f] * rpt / (MIN_PER_WEEK * util)
                            )
                if usage_terms:
                    model += (
                        pulp.lpSum(usage_terms) <= total_tools[q][ws][f],
                        f"cap_mt_{q}_{ws}_{f}",
                    )

    # C5. TOR tool capacity
    for q in QUARTERS:
        for ws in MINTECH_WS:
            ws_tor = TOR_MAP[ws]
            _, _, util_tor = WS_SPECS[ws_tor]
            for f in FABS:
                usage_terms = []
                for n in NODES:
                    for step_num, ws_step, rpt, ws_tor_step, rpt_tor in PROCESS_STEPS[
                        n
                    ]:
                        if ws_step == ws:
                            usage_terms.append(
                                flow_tor[q][n][step_num][f]
                                * rpt_tor
                                / (MIN_PER_WEEK * util_tor)
                            )
                if usage_terms:
                    model += (
                        pulp.lpSum(usage_terms) <= total_tools[q][ws_tor][f],
                        f"cap_tor_{q}_{ws_tor}_{f}",
                    )

    # C6. Fab space constraint
    for q in QUARTERS:
        for f in FABS:
            space_terms = []
            for ws in ALL_WS:
                space, _, _ = WS_SPECS[ws]
                space_terms.append(space * total_tools[q][ws][f])
            model += (
                pulp.lpSum(space_terms) <= FAB_SPECS[f]["space"],
                f"space_{q}_{f}",
            )

    # C7. No move-outs: total tools >= initial tools
    for q in QUARTERS:
        for ws in ALL_WS:
            for f in FABS:
                initial = FAB_SPECS[f]["tools"][ws]
                model += (total_tools[q][ws][f] >= initial, f"no_moveout_{q}_{ws}_{f}")

    # C8. Monotonicity: total tools non-decreasing over time
    for qi in range(1, len(QUARTERS)):
        q = QUARTERS[qi]
        q_prev = QUARTERS[qi - 1]
        for ws in ALL_WS:
            for f in FABS:
                model += (
                    total_tools[q][ws][f] >= total_tools[q_prev][ws][f],
                    f"monotone_{q}_{ws}_{f}",
                )

    return model, flow_mt, flow_tor, total_tools, transfer


# ─────────────────────────────────────────────────────────────────────────────
# SOLVE AND EXTRACT RESULTS
# ─────────────────────────────────────────────────────────────────────────────


def solve_and_report(model, flow_mt, flow_tor, total_tools, transfer):
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

    results = {
        "status": pulp.LpStatus[status],
        "total_cost": obj_val,
        "flow_mt": {},
        "flow_tor": {},
        "total_tools": {},
        "capex": 0,
        "opex_transfer": 0,
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
        results["total_tools"][q] = {}
        for ws in ALL_WS:
            results["total_tools"][q][ws] = {}
            for f in FABS:
                val = pulp.value(total_tools[q][ws][f])
                results["total_tools"][q][ws][f] = max(0, round(val)) if val else 0

    # CapEx
    capex = 0
    for qi, q in enumerate(QUARTERS):
        for ws in ALL_WS:
            _, cost, _ = WS_SPECS[ws]
            for f in FABS:
                initial = FAB_SPECS[f]["tools"][ws]
                if qi == 0:
                    bought = results["total_tools"][q][ws][f] - initial
                else:
                    q_prev = QUARTERS[qi - 1]
                    bought = (
                        results["total_tools"][q][ws][f]
                        - results["total_tools"][q_prev][ws][f]
                    )
                capex += max(0, bought) * cost
    results["capex"] = capex

    # OpEx
    opex = 0
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
                                opex += val * TRANSFER_COST * WEEKS_PER_Q
    results["opex_transfer"] = opex

    print(f"\n  CapEx (new tools):      ${capex:>15,.0f}")
    print(f"  OpEx (transfers):       ${opex:>15,.0f}")
    print(f"  Total Cost:             ${capex + opex:>15,.0f}")

    # Validate space constraints
    print("\n=== Space Validation ===")
    for q in QUARTERS:
        for f in FABS:
            used = sum(
                WS_SPECS[ws][0] * results["total_tools"][q][ws][f] for ws in ALL_WS
            )
            avail = FAB_SPECS[f]["space"]
            status_str = "OK" if used <= avail + 0.01 else "VIOLATED"
            if used > avail + 0.01:
                print(f"  {q} Fab {f}: {used:.1f}/{avail} m² [{status_str}]")

    # Print tool summary for Q1'26 and Q4'27
    for q in [QUARTERS[0], QUARTERS[-1]]:
        print(f"\n=== Tool Counts: {q} ===")
        for ws in ALL_WS:
            counts = [results["total_tools"][q][ws][f] for f in FABS]
            initial = [FAB_SPECS[f]["tools"][ws] for f in FABS]
            new = [counts[i] - initial[i] for i in range(3)]
            if any(counts[i] > 0 or new[i] != 0 for i in range(3)):
                print(
                    f"  {ws:3s}: F1={counts[0]:3d}(+{new[0]:3d}), F2={counts[1]:3d}(+{new[1]:3d}), F3={counts[2]:3d}(+{new[2]:3d})"
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
    model, flow_mt, flow_tor, total_tools, transfer = build_q1a_model()
    results = solve_and_report(model, flow_mt, flow_tor, total_tools, transfer)
    if results:
        save_results(results, "results/q1a_results.json")
        print("\nQ1a optimization complete.")
