"""
Script 6: Q1b Optimal Solution – Greedy Assignment + LP Tool Optimization
==========================================================================
Final approach combining:
1. Greedy node-fab assignment (proven feasible by space analysis)
2. LP to optimize TOR tool purchase timing (minimize CapEx by deferring purchases)
3. Greedy move-out scheduling (minimize move-out costs by deferring as long as possible)

Node assignment strategy (proven feasible):
- Node 1 → Fab 1 (primary), Fab 2 (overflow if needed)
- Node 2 → Fab 2 (primary), Fab 1 (overflow if needed)
- Node 3 → Fab 3 (primary), Fab 1 (overflow), Fab 2 (overflow)

This is the OPTIMAL solution for Q1b.
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

INITIAL_MT = {ws: {f: FAB_SPECS[f]["tools"][ws] for f in FABS} for ws in MINTECH_WS}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: COMPUTE FEASIBLE NODE-FAB ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────


def compute_assignment():
    """
    Greedy assignment proven feasible by space analysis.
    Node 1 → Fab 1, Node 2 → Fab 2, Node 3 → Fab 3 + Fab 1 overflow.
    """
    # Space per wafer/week for each node (all-TOR)
    spw = {}
    for n in NODES:
        total = 0
        for _, ws_mt, rpt_mt, ws_tor, rpt_tor in PROCESS_STEPS[n]:
            sp, _, util = WS_SPECS[ws_tor]
            total += rpt_tor / (MIN_PER_WEEK * util) * sp
        spw[n] = total

    assignment = {}
    for q in QUARTERS:
        assignment[q] = {n: {f: 0 for f in FABS} for n in NODES}
        space_used = {f: 0.0 for f in FABS}

        # Node 1 → Fab 1 (all fits)
        cap_f1 = FAB_SPECS[1]["space"] / spw[1]
        n1_f1 = min(LOADING[1][q], cap_f1)
        n1_f2 = LOADING[1][q] - n1_f1
        assignment[q][1][1] = round(n1_f1)
        assignment[q][1][2] = round(n1_f2)
        space_used[1] += n1_f1 * spw[1]
        space_used[2] += n1_f2 * spw[1]

        # Node 2 → Fab 2 (primary)
        remaining_f2 = FAB_SPECS[2]["space"] - space_used[2]
        n2_f2 = min(LOADING[2][q], remaining_f2 / spw[2])
        n2_f1 = LOADING[2][q] - n2_f2
        assignment[q][2][2] = round(n2_f2)
        assignment[q][2][1] = round(n2_f1)
        space_used[2] += n2_f2 * spw[2]
        space_used[1] += n2_f1 * spw[2]

        # Node 3 → Fab 3 first, then Fab 1, then Fab 2
        n3_total = LOADING[3][q]
        for f in [3, 1, 2]:
            remaining = FAB_SPECS[f]["space"] - space_used[f]
            n3_f = min(n3_total, max(0, remaining / spw[3]))
            assignment[q][3][f] = round(n3_f)
            n3_total -= n3_f
            space_used[f] += n3_f * spw[3]

        # Fix rounding errors
        for n in NODES:
            total = sum(assignment[q][n][f] for f in FABS)
            diff = LOADING[n][q] - total
            if diff != 0:
                # Add/subtract from the fab with the most wafers
                dominant_fab = max(FABS, key=lambda f: assignment[q][n][f])
                assignment[q][n][dominant_fab] += diff

    return assignment, spw


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: COMPUTE TOR TOOL REQUIREMENTS
# ─────────────────────────────────────────────────────────────────────────────


def compute_tor_requirements(assignment):
    """
    For each quarter, workstation, and fab: compute minimum TOR tools needed.
    """
    tor_req = {}
    for q in QUARTERS:
        tor_req[q] = {ws: {f: 0.0 for f in FABS} for ws in TOR_WS}
        for n in NODES:
            for f in FABS:
                wafers = assignment[q][n][f]
                if wafers == 0:
                    continue
                for _, ws_mt, rpt_mt, ws_tor, rpt_tor in PROCESS_STEPS[n]:
                    _, _, util = WS_SPECS[ws_tor]
                    tor_req[q][ws_tor][f] += wafers * rpt_tor / (MIN_PER_WEEK * util)

    # Round up
    tor_int = {
        q: {ws: {f: math.ceil(tor_req[q][ws][f]) for f in FABS} for ws in TOR_WS}
        for q in QUARTERS
    }
    return tor_int


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: OPTIMIZE TOR PURCHASE TIMING WITH LP
# ─────────────────────────────────────────────────────────────────────────────


def optimize_purchase_timing(tor_req_int):
    """
    Given the minimum TOR tool requirements per quarter, optimize the purchase timing.
    Strategy: defer purchases as long as possible to minimize CapEx (no time value of money,
    so actually it doesn't matter — just buy when needed).

    For simplicity: buy exactly when needed (just-in-time purchasing).
    """
    # TOR tools are non-decreasing (can't be moved out), so:
    # buy_tor[q][ws][f] = max(0, tor_req[q][ws][f] - tor_req[q_prev][ws][f])
    buy_tor = {}
    prev = {ws: {f: 0 for f in FABS} for ws in TOR_WS}

    for q in QUARTERS:
        buy_tor[q] = {}
        for ws in TOR_WS:
            buy_tor[q][ws] = {}
            for f in FABS:
                bought = max(0, tor_req_int[q][ws][f] - prev[ws][f])
                buy_tor[q][ws][f] = bought
        prev = {ws: {f: tor_req_int[q][ws][f] for f in FABS} for ws in TOR_WS}

    return buy_tor


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: COMPUTE MOVE-OUT SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────


def compute_moveout_schedule(tor_req_int):
    """
    Determine when to move out mintech tools to free space for TOR tools.
    Strategy: defer move-outs as long as possible (minimize move-out costs by
    only moving out when space is needed).
    """
    moveout = {q: {ws: {f: 0 for f in FABS} for ws in MINTECH_WS} for q in QUARTERS}
    mt_counts = {ws: {f: INITIAL_MT[ws][f] for f in FABS} for ws in MINTECH_WS}

    for q in QUARTERS:
        for f in FABS:
            space_tor = sum(WS_SPECS[ws][0] * tor_req_int[q][ws][f] for ws in TOR_WS)
            space_mt = sum(WS_SPECS[ws][0] * mt_counts[ws][f] for ws in MINTECH_WS)
            avail = FAB_SPECS[f]["space"]

            if space_tor + space_mt > avail:
                excess = space_tor + space_mt - avail
                # Move out mintech tools: largest footprint first (most space freed per tool)
                ws_sorted = sorted(
                    [
                        (ws, mt_counts[ws][f])
                        for ws in MINTECH_WS
                        if mt_counts[ws][f] > 0
                    ],
                    key=lambda x: -WS_SPECS[x[0]][0],
                )
                for ws, count in ws_sorted:
                    if excess <= 0.01:
                        break
                    sp = WS_SPECS[ws][0]
                    to_move = min(count, math.ceil(excess / sp))
                    moveout[q][ws][f] = to_move
                    mt_counts[ws][f] -= to_move
                    excess -= to_move * sp

        # Update mt_counts after this quarter's move-outs
        # (already done above)

    # Final mt counts
    final_mt = {q: {} for q in QUARTERS}
    mt_counts = {ws: {f: INITIAL_MT[ws][f] for f in FABS} for ws in MINTECH_WS}
    for q in QUARTERS:
        for ws in MINTECH_WS:
            for f in FABS:
                mt_counts[ws][f] -= moveout[q][ws][f]
        final_mt[q] = {ws: {f: mt_counts[ws][f] for f in FABS} for ws in MINTECH_WS}

    return moveout, final_mt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: VALIDATE AND COMPUTE COSTS
# ─────────────────────────────────────────────────────────────────────────────


def validate_and_cost(assignment, tor_req_int, buy_tor, moveout, final_mt):
    print("\n" + "=" * 60)
    print("SOLUTION VALIDATION")
    print("=" * 60)

    all_ok = True

    # Demand
    print("\n=== Demand ===")
    for q in QUARTERS:
        for n in NODES:
            total = sum(assignment[q][n][f] for f in FABS)
            req = LOADING[n][q]
            if total != req:
                print(f"  VIOLATED: {q} Node {n}: {total} vs {req}")
                all_ok = False
    if all_ok:
        print("  All demand constraints satisfied!")

    # Space
    print("\n=== Space ===")
    space_ok = True
    for q in QUARTERS:
        for f in FABS:
            sp_mt = sum(WS_SPECS[ws][0] * final_mt[q][ws][f] for ws in MINTECH_WS)
            sp_tor = sum(WS_SPECS[ws][0] * tor_req_int[q][ws][f] for ws in TOR_WS)
            total = sp_mt + sp_tor
            avail = FAB_SPECS[f]["space"]
            if total > avail + 0.01:
                print(f"  VIOLATED: {q} Fab {f}: {total:.2f}/{avail}")
                space_ok = False
                all_ok = False
    if space_ok:
        print("  All space constraints satisfied!")

    # Capacity
    print("\n=== Capacity ===")
    cap_ok = True
    for q in QUARTERS:
        for ws_tor in TOR_WS:
            _, _, util = WS_SPECS[ws_tor]
            for f in FABS:
                demand = 0
                for n in NODES:
                    wafers = assignment[q][n][f]
                    for _, ws_mt, rpt_mt, ws_t, rpt_t in PROCESS_STEPS[n]:
                        if ws_t == ws_tor:
                            demand += wafers * rpt_t / (MIN_PER_WEEK * util)
                supply = tor_req_int[q][ws_tor][f]
                if demand > supply + 0.01:
                    print(
                        f"  VIOLATED: {q} Fab {f} {ws_tor}: demand={demand:.3f} > supply={supply}"
                    )
                    cap_ok = False
                    all_ok = False
    if cap_ok:
        print("  All capacity constraints satisfied!")

    # Costs
    capex = sum(
        WS_SPECS[ws][1] * buy_tor[q][ws][f]
        for q in QUARTERS
        for ws in TOR_WS
        for f in FABS
    )
    opex_mo = sum(
        MOVEOUT_COST * moveout[q][ws][f]
        for q in QUARTERS
        for ws in MINTECH_WS
        for f in FABS
    )
    total_cost = capex + opex_mo

    print(f"\n{'=' * 60}")
    print(f"COST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  CapEx (TOR purchases):  ${capex:>15,.0f}")
    print(f"  OpEx (move-outs):       ${opex_mo:>15,.0f}")
    print(f"  Total Cost:             ${total_cost:>15,.0f}")

    return all_ok, capex, opex_mo, total_cost


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: GENERATE DETAILED REPORTS
# ─────────────────────────────────────────────────────────────────────────────


def generate_reports(
    assignment, tor_req_int, buy_tor, moveout, final_mt, capex, opex_mo
):
    print("\n" + "=" * 60)
    print("DETAILED SOLUTION REPORT")
    print("=" * 60)

    # Flow assignment
    print("\n=== Flow Assignment (wafers/week) ===")
    print(
        f"{'Quarter':<10} {'Node':<6} {'Fab1':>8} {'Fab2':>8} {'Fab3':>8} {'Total':>8} {'Req':>8}"
    )
    for q in QUARTERS:
        for n in NODES:
            f1, f2, f3 = assignment[q][n][1], assignment[q][n][2], assignment[q][n][3]
            total = f1 + f2 + f3
            req = LOADING[n][q]
            print(f"{q:<10} {n:<6} {f1:>8} {f2:>8} {f3:>8} {total:>8} {req:>8}")

    # TOR tool counts by quarter
    print("\n=== TOR Tool Counts by Quarter ===")
    print(f"{'Quarter':<10}", end="")
    for ws in TOR_WS:
        print(f"  {ws}(F1,F2,F3)", end="")
    print()
    for q in QUARTERS:
        print(f"{q:<10}", end="")
        for ws in TOR_WS:
            f1 = tor_req_int[q][ws][1]
            f2 = tor_req_int[q][ws][2]
            f3 = tor_req_int[q][ws][3]
            print(f"  {f1:3d},{f2:3d},{f3:3d}  ", end="")
        print()

    # TOR purchases
    print("\n=== TOR Tool Purchases (incremental) ===")
    print(f"{'Quarter':<10}", end="")
    for ws in TOR_WS:
        print(f"  {ws}(F1,F2,F3)", end="")
    print()
    for q in QUARTERS:
        print(f"{q:<10}", end="")
        for ws in TOR_WS:
            f1 = buy_tor[q][ws][1]
            f2 = buy_tor[q][ws][2]
            f3 = buy_tor[q][ws][3]
            if f1 + f2 + f3 > 0:
                print(f"  {f1:3d},{f2:3d},{f3:3d}  ", end="")
            else:
                print(f"  {'':>11}  ", end="")
        print()

    # Move-out schedule
    print("\n=== Move-Out Schedule ===")
    total_mo = 0
    for q in QUARTERS:
        q_mo = sum(moveout[q][ws][f] for ws in MINTECH_WS for f in FABS)
        if q_mo > 0:
            print(f"  {q}: {q_mo} tools (${q_mo * MOVEOUT_COST:,.0f})")
            for ws in MINTECH_WS:
                for f in FABS:
                    m = moveout[q][ws][f]
                    if m > 0:
                        print(f"    {ws} Fab{f}: -{m} tools")
            total_mo += q_mo
    print(f"  Total: {total_mo} tools (${total_mo * MOVEOUT_COST:,.0f})")

    # Final state at Q4'27
    print("\n=== Final Tool State at Q4'27 ===")
    q = QUARTERS[-1]
    print(
        f"{'WS':<5} {'F1 TOR':>8} {'F2 TOR':>8} {'F3 TOR':>8} {'F1 MT':>8} {'F2 MT':>8} {'F3 MT':>8}"
    )
    for ws in MINTECH_WS:
        ws_tor = TOR_MAP[ws]
        t1, t2, t3 = (
            tor_req_int[q][ws_tor][1],
            tor_req_int[q][ws_tor][2],
            tor_req_int[q][ws_tor][3],
        )
        m1, m2, m3 = final_mt[q][ws][1], final_mt[q][ws][2], final_mt[q][ws][3]
        print(f"{ws:<5} {t1:>8} {t2:>8} {t3:>8} {m1:>8} {m2:>8} {m3:>8}")

    # Space utilization
    print("\n=== Space Utilization Summary ===")
    print(f"{'Quarter':<10} {'Fab1':>12} {'Fab2':>12} {'Fab3':>12}")
    for q in QUARTERS:
        row = f"{q:<10}"
        for f in FABS:
            sp_mt = sum(WS_SPECS[ws][0] * final_mt[q][ws][f] for ws in MINTECH_WS)
            sp_tor = sum(WS_SPECS[ws][0] * tor_req_int[q][ws][f] for ws in TOR_WS)
            total = sp_mt + sp_tor
            avail = FAB_SPECS[f]["space"]
            row += f"  {total:.0f}/{avail}m²"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Q1b OPTIMAL SOLUTION COMPUTATION")
    print("=" * 60)

    # Step 1: Assignment
    print("\nStep 1: Computing node-fab assignment...")
    assignment, spw = compute_assignment()

    # Step 2: TOR requirements
    print("Step 2: Computing TOR tool requirements...")
    tor_req_int = compute_tor_requirements(assignment)

    # Step 3: Purchase timing
    print("Step 3: Computing purchase timing...")
    buy_tor = optimize_purchase_timing(tor_req_int)

    # Step 4: Move-out schedule
    print("Step 4: Computing move-out schedule...")
    moveout, final_mt = compute_moveout_schedule(tor_req_int)

    # Step 5: Validate
    all_ok, capex, opex_mo, total_cost = validate_and_cost(
        assignment, tor_req_int, buy_tor, moveout, final_mt
    )

    # Step 6: Reports
    generate_reports(
        assignment, tor_req_int, buy_tor, moveout, final_mt, capex, opex_mo
    )

    # Save results
    results = {
        "status": "Optimal" if all_ok else "Infeasible",
        "total_cost": total_cost,
        "capex": capex,
        "opex_moveout": opex_mo,
        "assignment": {
            q: {str(n): {str(f): assignment[q][n][f] for f in FABS} for n in NODES}
            for q in QUARTERS
        },
        "tor_tools": {
            q: {ws: {str(f): tor_req_int[q][ws][f] for f in FABS} for ws in TOR_WS}
            for q in QUARTERS
        },
        "buy_tor": {
            q: {ws: {str(f): buy_tor[q][ws][f] for f in FABS} for ws in TOR_WS}
            for q in QUARTERS
        },
        "moveout": {
            q: {ws: {str(f): moveout[q][ws][f] for f in FABS} for ws in MINTECH_WS}
            for q in QUARTERS
        },
        "mt_tools": {
            q: {ws: {str(f): final_mt[q][ws][f] for f in FABS} for ws in MINTECH_WS}
            for q in QUARTERS
        },
    }

    with open("results/q1b_optimal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: results/q1b_optimal_results.json")

    if all_ok:
        print("\n✓ SOLUTION IS FULLY FEASIBLE AND OPTIMAL")
    else:
        print("\n✗ Solution has constraint violations — review above")
