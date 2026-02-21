"""
Script 4: Q1b Heuristic-Guided MILP Solver
===========================================
Strategy: Fix node-to-fab assignment heuristically, then solve tool counts via LP.

Approach:
1. Assign each node primarily to the fab that already has the most relevant tools
   - Node 1 (A,D,F steps): Fab 1 or Fab 2 (both have A,D,F)
   - Node 2 (B,E,F steps): Fab 1 or Fab 2 (both have B,E,F)
   - Node 3 (C,D,E,F steps): Fab 3 (has C,D,E,F) + overflow to Fab 1/2
2. Allow cross-fab flow only when necessary (space overflow)
3. Solve tool counts as a pure LP (relax integrality) then round up

This dramatically reduces the search space and allows the solver to find
a feasible solution quickly.
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: HEURISTIC NODE-FAB ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────


def compute_node_fab_assignment():
    """
    Assign each node to fabs based on workstation compatibility and space.

    Strategy:
    - Node 1 (uses A, D, F): Assign 100% to Fab 1 (has A=50, D=50, F=90)
    - Node 2 (uses B, E, F): Assign 100% to Fab 2 (has B=30, E=30, F=60)
    - Node 3 (uses C, D, E, F): Split between Fab 3 (primary) and Fab 1/2 (overflow)
      - Fab 3 has C=40, D=35, E=16, F=36 (limited capacity)
      - Overflow goes to Fab 1 (has D, E, F) and Fab 2 (has D, E, F)
      - For C workstation overflow: must purchase C tools in Fab 1 or Fab 2

    Returns: dict[q][n][f] = fraction of loading assigned to fab f
    """
    assignment = {}

    for q in QUARTERS:
        assignment[q] = {}

        # Node 1: Fab 1 primary, Fab 2 secondary
        # At peak (Q1'26 = 12000), Fab 1 can handle it
        assignment[q][1] = {1: 1.0, 2: 0.0, 3: 0.0}

        # Node 2: Fab 2 primary, Fab 1 secondary
        assignment[q][2] = {1: 0.0, 2: 1.0, 3: 0.0}

        # Node 3: Fab 3 primary, overflow to Fab 1 and Fab 2
        # Fab 3 space: 700 m², used: 636 m², available: 64 m²
        # At Q4'27 (16000 wafers/week), we need ~157 C+ tools = 903 m²
        # Must spread across all 3 fabs
        # Rough split: Fab3=30%, Fab1=35%, Fab2=35%
        loading_n3 = LOADING[3][q]
        if loading_n3 <= 3000:
            assignment[q][3] = {1: 0.0, 2: 0.0, 3: 1.0}
        elif loading_n3 <= 6000:
            assignment[q][3] = {1: 0.0, 2: 0.0, 3: 1.0}
        elif loading_n3 <= 9000:
            assignment[q][3] = {1: 0.1, 2: 0.1, 3: 0.8}
        elif loading_n3 <= 12000:
            assignment[q][3] = {1: 0.2, 2: 0.2, 3: 0.6}
        else:
            # Q3'27 (13000) and Q4'27 (16000)
            assignment[q][3] = {1: 0.35, 2: 0.35, 3: 0.30}

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: COMPUTE REQUIRED TOOL COUNTS GIVEN ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────


def compute_tool_requirements(assignment):
    """
    Given a node-fab assignment, compute the minimum tool counts needed.
    Uses TOR tools preferentially to minimize space usage.

    Returns: dict[q][ws][f] = tool count (TOR preferred)
    """
    # For each quarter, workstation, fab: compute tool requirement
    req = {}
    for q in QUARTERS:
        req[q] = {}
        for ws in ALL_WS:
            req[q][ws] = {f: 0.0 for f in FABS}

        for n in NODES:
            loading_n = LOADING[n][q]
            for f in FABS:
                frac = assignment[q][n][f]
                if frac <= 0:
                    continue
                wafers_in_fab = loading_n * frac

                for step_num, ws_mt, rpt_mt, ws_tor, rpt_tor in PROCESS_STEPS[n]:
                    # Use TOR tools preferentially (better utilization, smaller space per throughput)
                    _, _, util_tor = WS_SPECS[ws_tor]
                    tool_req_tor = wafers_in_fab * rpt_tor / (MIN_PER_WEEK * util_tor)
                    req[q][ws_tor][f] += tool_req_tor

    # Round up
    req_int = {}
    for q in QUARTERS:
        req_int[q] = {}
        for ws in ALL_WS:
            req_int[q][ws] = {}
            for f in FABS:
                req_int[q][ws][f] = math.ceil(req[q][ws][f])

    return req_int


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: CHECK SPACE FEASIBILITY AND ADJUST
# ─────────────────────────────────────────────────────────────────────────────


def check_space_feasibility(req_int):
    """Check if the tool requirements fit within fab space constraints."""
    print("\n=== Space Feasibility Check ===")
    all_ok = True
    for q in QUARTERS:
        for f in FABS:
            # Space used by initial mintech tools that are still present
            # (we'll move out as needed)
            # For now, assume we keep initial tools and add TOR
            space_initial = sum(
                WS_SPECS[ws][0] * FAB_SPECS[f]["tools"][ws] for ws in MINTECH_WS
            )
            space_new_tor = sum(WS_SPECS[ws][0] * req_int[q][ws][f] for ws in TOR_WS)
            total_space = space_initial + space_new_tor
            avail = FAB_SPECS[f]["space"]

            if total_space > avail:
                # Need to move out some initial mintech tools
                excess = total_space - avail
                print(
                    f"  {q} Fab {f}: Need to free {excess:.1f} m² (total={total_space:.1f}, avail={avail})"
                )
                all_ok = False

    if all_ok:
        print("  All space constraints satisfied with initial tools + new TOR!")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: FULL LP MODEL WITH FIXED ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────


def build_lp_model(assignment):
    """
    Build an LP model with fixed node-fab assignment fractions.
    Only optimizes tool counts (mintech vs TOR mix) and move-outs.
    """
    print("\nBuilding LP model with fixed node-fab assignment...")
    model = pulp.LpProblem("Q1b_LP_Fixed_Assignment", pulp.LpMinimize)

    # Tool count variables (continuous relaxation for speed)
    # total_mt[q][ws][f]: mintech tools
    # total_tor[q][ws][f]: TOR tools
    total_mt = {
        q: {
            ws: {f: pulp.LpVariable(f"mt_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in MINTECH_WS
        }
        for q in QUARTERS
    }
    total_tor = {
        q: {
            ws: {f: pulp.LpVariable(f"tor_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in TOR_WS
        }
        for q in QUARTERS
    }

    # Purchase and move-out variables
    purchase_mt = {
        q: {
            ws: {f: pulp.LpVariable(f"buy_mt_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in MINTECH_WS
        }
        for q in QUARTERS
    }
    moveout_mt = {
        q: {
            ws: {f: pulp.LpVariable(f"out_mt_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in MINTECH_WS
        }
        for q in QUARTERS
    }
    purchase_tor = {
        q: {
            ws: {f: pulp.LpVariable(f"buy_tor_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in TOR_WS
        }
        for q in QUARTERS
    }

    # Link variables
    for qi, q in enumerate(QUARTERS):
        for ws in MINTECH_WS:
            for f in FABS:
                initial = FAB_SPECS[f]["tools"][ws]
                if qi == 0:
                    model += (
                        total_mt[q][ws][f]
                        == initial + purchase_mt[q][ws][f] - moveout_mt[q][ws][f]
                    )
                else:
                    q_prev = QUARTERS[qi - 1]
                    model += (
                        total_mt[q][ws][f]
                        == total_mt[q_prev][ws][f]
                        + purchase_mt[q][ws][f]
                        - moveout_mt[q][ws][f]
                    )
        for ws in TOR_WS:
            for f in FABS:
                if qi == 0:
                    model += total_tor[q][ws][f] == purchase_tor[q][ws][f]
                else:
                    q_prev = QUARTERS[qi - 1]
                    model += (
                        total_tor[q][ws][f]
                        == total_tor[q_prev][ws][f] + purchase_tor[q][ws][f]
                    )

    # Objective
    capex_terms = []
    for q in QUARTERS:
        for ws in MINTECH_WS:
            _, capex, _ = WS_SPECS[ws]
            for f in FABS:
                capex_terms.append(capex * purchase_mt[q][ws][f])
        for ws in TOR_WS:
            _, capex, _ = WS_SPECS[ws]
            for f in FABS:
                capex_terms.append(capex * purchase_tor[q][ws][f])

    moveout_terms = [
        MOVEOUT_COST * moveout_mt[q][ws][f]
        for q in QUARTERS
        for ws in MINTECH_WS
        for f in FABS
    ]

    # Transfer cost (based on fixed assignment, cross-fab wafers)
    transfer_cost_total = 0
    for q in QUARTERS:
        for n in NODES:
            loading_n = LOADING[n][q]
            steps = PROCESS_STEPS[n]
            for idx in range(1, len(steps)):
                for f_from in FABS:
                    for f_to in FABS:
                        if f_from != f_to:
                            # Approximate: if assignment changes between steps, there's a transfer
                            # For fixed assignment, no transfers within a node
                            pass

    model += pulp.lpSum(capex_terms) + pulp.lpSum(moveout_terms), "Total_Cost"

    # Capacity constraints
    for q in QUARTERS:
        for ws in MINTECH_WS:
            _, _, util = WS_SPECS[ws]
            ws_tor = TOR_MAP[ws]
            _, _, util_tor = WS_SPECS[ws_tor]
            for f in FABS:
                # Compute required throughput for this ws in this fab
                mt_usage = []
                tor_usage = []
                for n in NODES:
                    frac = assignment[q][n][f]
                    if frac <= 0:
                        continue
                    wafers = LOADING[n][q] * frac
                    for (
                        step_num,
                        ws_step,
                        rpt_mt,
                        ws_tor_step,
                        rpt_tor,
                    ) in PROCESS_STEPS[n]:
                        if ws_step == ws:
                            mt_usage.append(wafers * rpt_mt / (MIN_PER_WEEK * util))
                            tor_usage.append(
                                wafers * rpt_tor / (MIN_PER_WEEK * util_tor)
                            )

                if mt_usage:
                    # Total throughput must be met by mt + tor tools
                    total_throughput_mt = sum(mt_usage)
                    total_throughput_tor = sum(tor_usage)
                    # Let x = fraction on mintech, (1-x) = fraction on TOR
                    # x * total_throughput_mt <= total_mt[q][ws][f]
                    # (1-x) * total_throughput_tor <= total_tor[q][ws_tor][f]
                    # Instead, use a combined constraint:
                    # total_mt + total_tor (in equivalent units) >= requirement
                    # We use the simpler: mt_req + tor_req >= total_requirement
                    # where mt_req is the tool count if all on mintech
                    # and tor_req is the tool count if all on TOR
                    # The actual constraint: mt_tools >= mt_fraction * mt_req
                    #                        tor_tools >= tor_fraction * tor_req
                    # For simplicity, require that combined capacity covers demand:
                    # (total_mt / mt_req_full) + (total_tor / tor_req_full) >= 1
                    # This is non-linear. Use linear approximation:
                    # Require total_mt >= 0 and total_tor >= demand_tor (all on TOR)
                    # This is conservative but feasible.
                    demand_tor = math.ceil(sum(tor_usage))
                    model += (
                        total_tor[q][ws_tor][f] >= demand_tor,
                        f"cap_tor_{q}_{ws_tor}_{f}",
                    )

    # Space constraints
    for q in QUARTERS:
        for f in FABS:
            space_terms = []
            for ws in MINTECH_WS:
                space, _, _ = WS_SPECS[ws]
                space_terms.append(space * total_mt[q][ws][f])
            for ws in TOR_WS:
                space, _, _ = WS_SPECS[ws]
                space_terms.append(space * total_tor[q][ws][f])
            model += pulp.lpSum(space_terms) <= FAB_SPECS[f]["space"], f"space_{q}_{f}"

    return model, total_mt, total_tor, purchase_mt, moveout_mt, purchase_tor


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: DIRECT COMPUTATION (NO LP NEEDED)
# ─────────────────────────────────────────────────────────────────────────────


def compute_optimal_solution():
    """
    Direct computation approach:
    1. Fix node-fab assignment
    2. For each quarter and fab, compute minimum tool counts using all-TOR strategy
    3. Determine move-outs needed to free space
    4. Compute costs
    """
    print("\n" + "=" * 60)
    print("DIRECT COMPUTATION APPROACH")
    print("=" * 60)

    assignment = compute_node_fab_assignment()

    # For each quarter, compute TOR tool requirements per fab
    results = {
        "assignment": {},
        "total_mt": {},
        "total_tor": {},
        "moveout_mt": {},
        "purchase_tor": {},
        "capex": 0,
        "opex_moveout": 0,
        "opex_transfer": 0,
    }

    # Convert assignment to integer wafer counts
    for q in QUARTERS:
        results["assignment"][q] = {}
        for n in NODES:
            results["assignment"][q][n] = {}
            for f in FABS:
                results["assignment"][q][n][f] = round(
                    LOADING[n][q] * assignment[q][n][f]
                )

    # Compute TOR tool requirements
    tor_req = {}
    for q in QUARTERS:
        tor_req[q] = {ws: {f: 0.0 for f in FABS} for ws in TOR_WS}
        for n in NODES:
            for f in FABS:
                wafers = results["assignment"][q][n][f]
                if wafers == 0:
                    continue
                for step_num, ws_mt, rpt_mt, ws_tor, rpt_tor in PROCESS_STEPS[n]:
                    _, _, util_tor = WS_SPECS[ws_tor]
                    tor_req[q][ws_tor][f] += (
                        wafers * rpt_tor / (MIN_PER_WEEK * util_tor)
                    )

    # Round up TOR requirements
    for q in QUARTERS:
        results["total_tor"][q] = {}
        for ws in TOR_WS:
            results["total_tor"][q][ws] = {}
            for f in FABS:
                results["total_tor"][q][ws][f] = math.ceil(tor_req[q][ws][f])

    # Determine move-outs needed to free space
    # Strategy: move out mintech tools that are no longer needed (replaced by TOR)
    # Keep mintech tools only if they help (i.e., there's remaining capacity after TOR)

    # First, compute space used by TOR tools at each quarter
    # Then determine how many mintech tools can remain

    results["total_mt"] = {}
    results["moveout_mt"] = {}

    # Track cumulative state
    current_mt = {ws: {f: FAB_SPECS[f]["tools"][ws] for f in FABS} for ws in MINTECH_WS}

    for q in QUARTERS:
        results["total_mt"][q] = {}
        results["moveout_mt"][q] = {}

        for ws in MINTECH_WS:
            results["total_mt"][q][ws] = {}
            results["moveout_mt"][q][ws] = {}
            for f in FABS:
                results["total_mt"][q][ws][f] = current_mt[ws][f]
                results["moveout_mt"][q][ws][f] = 0

        # Check space for each fab
        for f in FABS:
            space_avail = FAB_SPECS[f]["space"]

            # Space used by TOR tools this quarter
            space_tor = sum(
                WS_SPECS[ws][0] * results["total_tor"][q][ws][f] for ws in TOR_WS
            )

            # Space used by current mintech tools
            space_mt = sum(WS_SPECS[ws][0] * current_mt[ws][f] for ws in MINTECH_WS)

            total_space = space_tor + space_mt

            if total_space > space_avail:
                # Need to move out mintech tools
                excess = total_space - space_avail

                # Move out mintech tools in order of least useful first
                # Priority: move out tools that are not needed for any node in this fab
                # For this fab, which workstations are needed?
                needed_ws = set()
                for n in NODES:
                    if results["assignment"][q][n][f] > 0:
                        for step_num, ws_mt, rpt_mt, ws_tor, rpt_tor in PROCESS_STEPS[
                            n
                        ]:
                            needed_ws.add(ws_mt)

                # Move out tools not needed first, then least-needed
                # Sort by: not needed > low count > high space
                ws_priority = []
                for ws in MINTECH_WS:
                    count = current_mt[ws][f]
                    if count == 0:
                        continue
                    needed = ws in needed_ws
                    space_per_tool = WS_SPECS[ws][0]
                    ws_priority.append((not needed, -space_per_tool, ws, count))

                ws_priority.sort(reverse=True)

                for not_needed, neg_space, ws, count in ws_priority:
                    if excess <= 0:
                        break
                    space_per_tool = WS_SPECS[ws][0]
                    # Move out as many as needed
                    tools_to_move = min(count, math.ceil(excess / space_per_tool))
                    if tools_to_move > 0:
                        results["moveout_mt"][q][ws][f] = tools_to_move
                        results["total_mt"][q][ws][f] = (
                            current_mt[ws][f] - tools_to_move
                        )
                        excess -= tools_to_move * space_per_tool

        # Update current_mt for next quarter
        for ws in MINTECH_WS:
            for f in FABS:
                current_mt[ws][f] = results["total_mt"][q][ws][f]

    # Compute purchase_tor (incremental)
    results["purchase_tor"] = {}
    prev_tor = {ws: {f: 0 for f in FABS} for ws in TOR_WS}
    for q in QUARTERS:
        results["purchase_tor"][q] = {}
        for ws in TOR_WS:
            results["purchase_tor"][q][ws] = {}
            for f in FABS:
                bought = results["total_tor"][q][ws][f] - prev_tor[ws][f]
                results["purchase_tor"][q][ws][f] = max(0, bought)
        prev_tor = {
            ws: {f: results["total_tor"][q][ws][f] for f in FABS} for ws in TOR_WS
        }

    # Compute costs
    capex = 0
    for q in QUARTERS:
        for ws in TOR_WS:
            _, cost, _ = WS_SPECS[ws]
            for f in FABS:
                capex += results["purchase_tor"][q][ws][f] * cost
    results["capex"] = capex

    opex_m = 0
    for q in QUARTERS:
        for ws in MINTECH_WS:
            for f in FABS:
                opex_m += results["moveout_mt"][q][ws][f] * MOVEOUT_COST
    results["opex_moveout"] = opex_m

    # Validate space
    print("\n=== Space Validation ===")
    all_ok = True
    for q in QUARTERS:
        for f in FABS:
            space_mt = sum(
                WS_SPECS[ws][0] * results["total_mt"][q][ws][f] for ws in MINTECH_WS
            )
            space_tor = sum(
                WS_SPECS[ws][0] * results["total_tor"][q][ws][f] for ws in TOR_WS
            )
            total = space_mt + space_tor
            avail = FAB_SPECS[f]["space"]
            if total > avail + 0.1:
                print(f"  VIOLATED: {q} Fab {f}: {total:.1f}/{avail} m²")
                all_ok = False
    if all_ok:
        print("  All space constraints satisfied!")

    # Validate capacity
    print("\n=== Capacity Validation ===")
    cap_ok = True
    for q in QUARTERS:
        for f in FABS:
            for ws in TOR_WS:
                ws_mt = ws[:-1]  # remove "+"
                _, _, util_tor = WS_SPECS[ws]
                demand = 0
                for n in NODES:
                    wafers = results["assignment"][q][n][f]
                    if wafers == 0:
                        continue
                    for (
                        step_num,
                        ws_step,
                        rpt_mt,
                        ws_tor_step,
                        rpt_tor,
                    ) in PROCESS_STEPS[n]:
                        if ws_tor_step == ws:
                            demand += wafers * rpt_tor / (MIN_PER_WEEK * util_tor)
                supply = results["total_tor"][q][ws][f]
                if demand > supply + 0.01:
                    print(
                        f"  VIOLATED: {q} Fab {f} {ws}: demand={demand:.2f} > supply={supply}"
                    )
                    cap_ok = False
    if cap_ok:
        print("  All capacity constraints satisfied!")

    print(f"\n  CapEx (TOR purchases):  ${capex:>15,.0f}")
    print(f"  OpEx (move-outs):       ${opex_m:>15,.0f}")
    print(f"  Total Cost:             ${capex + opex_m:>15,.0f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: OPTIMIZE ASSIGNMENT WITH LP
# ─────────────────────────────────────────────────────────────────────────────


def optimize_with_lp():
    """
    Full LP optimization: continuous relaxation of tool counts.
    Fixes the structure (all-TOR strategy) but optimizes the node-fab assignment
    and tool counts simultaneously.
    """
    print("\n" + "=" * 60)
    print("LP OPTIMIZATION (CONTINUOUS RELAXATION)")
    print("=" * 60)

    model = pulp.LpProblem("Q1b_LP_Optimization", pulp.LpMinimize)

    # Flow variables: wafers of node n processed in fab f in quarter q
    # (all on TOR tools for simplicity)
    flow = {
        q: {
            n: {f: pulp.LpVariable(f"flow_{q}_{n}_{f}", lowBound=0) for f in FABS}
            for n in NODES
        }
        for q in QUARTERS
    }

    # TOR tool counts (continuous)
    total_tor = {
        q: {
            ws: {f: pulp.LpVariable(f"tor_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in TOR_WS
        }
        for q in QUARTERS
    }

    # Mintech tool counts (continuous, can decrease via move-outs)
    total_mt = {
        q: {
            ws: {f: pulp.LpVariable(f"mt_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in MINTECH_WS
        }
        for q in QUARTERS
    }

    # Purchase and move-out variables
    purchase_tor = {
        q: {
            ws: {f: pulp.LpVariable(f"buy_tor_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in TOR_WS
        }
        for q in QUARTERS
    }
    moveout_mt = {
        q: {
            ws: {f: pulp.LpVariable(f"out_mt_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in MINTECH_WS
        }
        for q in QUARTERS
    }

    # Link constraints
    for qi, q in enumerate(QUARTERS):
        for ws in MINTECH_WS:
            for f in FABS:
                initial = FAB_SPECS[f]["tools"][ws]
                if qi == 0:
                    model += total_mt[q][ws][f] == initial - moveout_mt[q][ws][f]
                else:
                    q_prev = QUARTERS[qi - 1]
                    model += (
                        total_mt[q][ws][f]
                        == total_mt[q_prev][ws][f] - moveout_mt[q][ws][f]
                    )
        for ws in TOR_WS:
            for f in FABS:
                if qi == 0:
                    model += total_tor[q][ws][f] == purchase_tor[q][ws][f]
                else:
                    q_prev = QUARTERS[qi - 1]
                    model += (
                        total_tor[q][ws][f]
                        == total_tor[q_prev][ws][f] + purchase_tor[q][ws][f]
                    )

    # Objective
    capex_terms = [
        WS_SPECS[ws][1] * purchase_tor[q][ws][f]
        for q in QUARTERS
        for ws in TOR_WS
        for f in FABS
    ]
    moveout_terms = [
        MOVEOUT_COST * moveout_mt[q][ws][f]
        for q in QUARTERS
        for ws in MINTECH_WS
        for f in FABS
    ]
    # Transfer cost: penalize cross-fab flow (approximate)
    # We don't model explicit transfers here, just penalize assignment changes

    model += pulp.lpSum(capex_terms) + pulp.lpSum(moveout_terms), "Total_Cost"

    # Demand constraints
    for q in QUARTERS:
        for n in NODES:
            model += (
                pulp.lpSum(flow[q][n][f] for f in FABS) == LOADING[n][q],
                f"demand_{q}_{n}",
            )

    # Capacity constraints (TOR tools)
    for q in QUARTERS:
        for ws in TOR_WS:
            ws_mt = ws[:-1]
            _, _, util_tor = WS_SPECS[ws]
            for f in FABS:
                usage = []
                for n in NODES:
                    for (
                        step_num,
                        ws_step,
                        rpt_mt,
                        ws_tor_step,
                        rpt_tor,
                    ) in PROCESS_STEPS[n]:
                        if ws_tor_step == ws:
                            usage.append(
                                flow[q][n][f] * rpt_tor / (MIN_PER_WEEK * util_tor)
                            )
                if usage:
                    model += (
                        pulp.lpSum(usage) <= total_tor[q][ws][f],
                        f"cap_tor_{q}_{ws}_{f}",
                    )

    # Space constraints
    for q in QUARTERS:
        for f in FABS:
            space_terms = [
                WS_SPECS[ws][0] * total_mt[q][ws][f] for ws in MINTECH_WS
            ] + [WS_SPECS[ws][0] * total_tor[q][ws][f] for ws in TOR_WS]
            model += pulp.lpSum(space_terms) <= FAB_SPECS[f]["space"], f"space_{q}_{f}"

    print(
        f"Model: {len(model.variables())} variables, {len(model.constraints)} constraints."
    )
    print("Solving LP relaxation...")

    start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)
    status = model.solve(solver)
    elapsed = time.time() - start

    print(f"Status: {pulp.LpStatus[status]}, Time: {elapsed:.1f}s")

    if pulp.value(model.objective) is None:
        print("LP infeasible!")
        return None

    obj = pulp.value(model.objective)
    print(f"LP Objective: ${obj:,.0f}")

    # Extract results
    results = {
        "status": pulp.LpStatus[status],
        "total_cost": obj,
        "flow": {},
        "total_mt": {},
        "total_tor": {},
        "moveout_mt": {},
        "purchase_tor": {},
        "capex": 0,
        "opex_moveout": 0,
        "opex_transfer": 0,
    }

    for q in QUARTERS:
        results["flow"][q] = {}
        for n in NODES:
            results["flow"][q][n] = {}
            for f in FABS:
                v = pulp.value(flow[q][n][f])
                results["flow"][q][n][f] = max(0, v) if v else 0

        results["total_mt"][q] = {}
        results["total_tor"][q] = {}
        results["moveout_mt"][q] = {}
        results["purchase_tor"][q] = {}

        for ws in MINTECH_WS:
            results["total_mt"][q][ws] = {}
            results["moveout_mt"][q][ws] = {}
            for f in FABS:
                v = pulp.value(total_mt[q][ws][f])
                results["total_mt"][q][ws][f] = max(0, v) if v else 0
                vm = pulp.value(moveout_mt[q][ws][f])
                results["moveout_mt"][q][ws][f] = max(0, vm) if vm else 0

        for ws in TOR_WS:
            results["total_tor"][q][ws] = {}
            results["purchase_tor"][q][ws] = {}
            for f in FABS:
                v = pulp.value(total_tor[q][ws][f])
                results["total_tor"][q][ws][f] = max(0, v) if v else 0
                vp = pulp.value(purchase_tor[q][ws][f])
                results["purchase_tor"][q][ws][f] = max(0, vp) if vp else 0

    # Round up tool counts to integers
    for q in QUARTERS:
        for ws in TOR_WS:
            for f in FABS:
                results["total_tor"][q][ws][f] = math.ceil(
                    results["total_tor"][q][ws][f]
                )
        for ws in MINTECH_WS:
            for f in FABS:
                results["total_mt"][q][ws][f] = math.floor(
                    results["total_mt"][q][ws][f]
                )
                results["moveout_mt"][q][ws][f] = math.ceil(
                    results["moveout_mt"][q][ws][f]
                )

    # Recompute purchase_tor after rounding
    prev_tor = {ws: {f: 0 for f in FABS} for ws in TOR_WS}
    for q in QUARTERS:
        for ws in TOR_WS:
            for f in FABS:
                bought = results["total_tor"][q][ws][f] - prev_tor[ws][f]
                results["purchase_tor"][q][ws][f] = max(0, bought)
        prev_tor = {
            ws: {f: results["total_tor"][q][ws][f] for f in FABS} for ws in TOR_WS
        }

    # Compute costs
    capex = sum(
        WS_SPECS[ws][1] * results["purchase_tor"][q][ws][f]
        for q in QUARTERS
        for ws in TOR_WS
        for f in FABS
    )
    opex_m = sum(
        MOVEOUT_COST * results["moveout_mt"][q][ws][f]
        for q in QUARTERS
        for ws in MINTECH_WS
        for f in FABS
    )
    results["capex"] = capex
    results["opex_moveout"] = opex_m

    print(f"\n  CapEx (TOR purchases):  ${capex:>15,.0f}")
    print(f"  OpEx (move-outs):       ${opex_m:>15,.0f}")
    print(f"  Total Cost:             ${capex + opex_m:>15,.0f}")

    # Validate space
    print("\n=== Space Validation ===")
    all_ok = True
    for q in QUARTERS:
        for f in FABS:
            space_mt = sum(
                WS_SPECS[ws][0] * results["total_mt"][q][ws][f] for ws in MINTECH_WS
            )
            space_tor = sum(
                WS_SPECS[ws][0] * results["total_tor"][q][ws][f] for ws in TOR_WS
            )
            total = space_mt + space_tor
            avail = FAB_SPECS[f]["space"]
            if total > avail + 0.1:
                print(f"  VIOLATED: {q} Fab {f}: {total:.1f}/{avail} m²")
                all_ok = False
    if all_ok:
        print("  All space constraints satisfied!")

    # Print flow assignment
    print("\n=== Flow Assignment (wafers/week) ===")
    print(
        f"{'Quarter':<10} {'Node':<6} {'Fab1':>8} {'Fab2':>8} {'Fab3':>8} {'Total':>8}"
    )
    for q in QUARTERS:
        for n in NODES:
            f1 = results["flow"][q][n][1]
            f2 = results["flow"][q][n][2]
            f3 = results["flow"][q][n][3]
            total = f1 + f2 + f3
            req = LOADING[n][q]
            ok = "OK" if abs(total - req) < 1 else f"ERR(need {req})"
            print(f"{q:<10} {n:<6} {f1:>8.0f} {f2:>8.0f} {f3:>8.0f} {total:>8.0f} {ok}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Q1b HEURISTIC SOLVER")
    print("=" * 60)

    # Approach 1: Direct computation
    results_direct = compute_optimal_solution()

    # Approach 2: LP optimization
    results_lp = optimize_with_lp()

    # Use LP results as primary (better optimization)
    if results_lp and results_lp["status"] == "Optimal":
        primary = results_lp
        print("\nUsing LP results as primary solution.")
    else:
        primary = results_direct
        print("\nUsing direct computation results as primary solution.")

    # Save results
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    with open("results/q1b_heuristic_results.json", "w") as f:
        json.dump(convert(primary), f, indent=2)
    print("\nResults saved to: results/q1b_heuristic_results.json")

    # Also save LP results separately
    if results_lp:
        with open("results/q1b_lp_results.json", "w") as f:
            json.dump(convert(results_lp), f, indent=2)
        print("LP results saved to: results/q1b_lp_results.json")
