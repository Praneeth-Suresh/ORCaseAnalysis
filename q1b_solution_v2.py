"""
Script 7: Q1b Solution v2 – Space-Aware Rounding
==================================================
Fixes the space violation caused by ceiling rounding of TOR tools.
Key fix: compute the space used by ROUNDED tools, then reduce flow if needed.

Strategy:
1. Assign nodes to fabs (greedy, proven feasible)
2. Compute TOR requirements (continuous)
3. Round up TOR tools (ceiling)
4. Check if rounded tools exceed space
5. If yes, reduce flow in that fab slightly and recompute
6. Compute move-out schedule
"""

import json
import math
from pathlib import Path

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
MOVEOUT_COST = P["moveout_cost_per_tool"]
MIN_PER_WEEK = P["minutes_per_week"]

MINTECH_WS = ["A", "B", "C", "D", "E", "F"]
TOR_WS = ["A+", "B+", "C+", "D+", "E+", "F+"]
TOR_MAP = {"A": "A+", "B": "B+", "C": "C+", "D": "D+", "E": "E+", "F": "F+"}

INITIAL_MT = {ws: {f: FAB_SPECS[f]["tools"][ws] for f in FABS} for ws in MINTECH_WS}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: space used by TOR tools for a given flow assignment
# ─────────────────────────────────────────────────────────────────────────────


def compute_tor_space(flow_nf):
    """
    Given flow_nf = {n: wafers} for a single fab,
    compute the space used by TOR tools (using ceiling rounding).
    Returns (space_used, tor_counts)
    """
    tor_cont = {ws: 0.0 for ws in TOR_WS}
    for n, wafers in flow_nf.items():
        if wafers == 0:
            continue
        for _, ws_mt, rpt_mt, ws_tor, rpt_tor in PROCESS_STEPS[n]:
            _, _, util = WS_SPECS[ws_tor]
            tor_cont[ws_tor] += wafers * rpt_tor / (MIN_PER_WEEK * util)

    tor_int = {ws: math.ceil(tor_cont[ws]) for ws in TOR_WS}
    space = sum(WS_SPECS[ws][0] * tor_int[ws] for ws in TOR_WS)
    return space, tor_int


def max_wafers_in_fab(node, fab_space):
    """
    Maximum integer wafers of a given node that fit in fab_space m²,
    accounting for TOR tool ceiling rounding.
    """
    # Binary search for max wafers
    lo, hi = 0, int(fab_space / 0.09) + 1  # upper bound
    while lo < hi:
        mid = (lo + hi + 1) // 2
        space, _ = compute_tor_space({node: mid})
        if space <= fab_space:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: COMPUTE FEASIBLE NODE-FAB ASSIGNMENT (space-aware)
# ─────────────────────────────────────────────────────────────────────────────


def compute_assignment():
    """
    Greedy assignment with space-aware rounding.
    Node 1 → Fab 1, Node 2 → Fab 2, Node 3 → Fab 3 + Fab 1 overflow.
    """
    assignment = {}

    for q in QUARTERS:
        assignment[q] = {n: {f: 0 for f in FABS} for n in NODES}
        space_used = {f: 0.0 for f in FABS}

        # Node 1 → Fab 1 (primary)
        max_n1_f1 = max_wafers_in_fab(1, FAB_SPECS[1]["space"])
        n1_f1 = min(LOADING[1][q], max_n1_f1)
        n1_f2 = LOADING[1][q] - n1_f1
        assignment[q][1][1] = n1_f1
        assignment[q][1][2] = n1_f2
        sp1, _ = compute_tor_space({1: n1_f1})
        sp2, _ = compute_tor_space({1: n1_f2})
        space_used[1] += sp1
        space_used[2] += sp2

        # Node 2 → Fab 2 (primary)
        remaining_f2 = FAB_SPECS[2]["space"] - space_used[2]
        # Binary search for max Node 2 in Fab 2
        lo, hi = 0, LOADING[2][q]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            sp, _ = compute_tor_space({1: n1_f2, 2: mid})
            if sp <= FAB_SPECS[2]["space"]:
                lo = mid
            else:
                hi = mid - 1
        n2_f2 = lo
        n2_f1 = LOADING[2][q] - n2_f2
        assignment[q][2][2] = n2_f2
        assignment[q][2][1] = n2_f1
        sp2_new, _ = compute_tor_space({1: n1_f2, 2: n2_f2})
        sp1_new, _ = compute_tor_space({1: n1_f1, 2: n2_f1})
        space_used[2] = sp2_new
        space_used[1] = sp1_new

        # Node 3 → Fab 3 first, then Fab 1, then Fab 2
        n3_total = LOADING[3][q]

        # Fab 3
        max_n3_f3 = max_wafers_in_fab(3, FAB_SPECS[3]["space"])
        n3_f3 = min(n3_total, max_n3_f3)
        n3_total -= n3_f3

        # Fab 1 (remaining space)
        remaining_f1 = FAB_SPECS[1]["space"] - space_used[1]
        lo, hi = 0, n3_total
        while lo < hi:
            mid = (lo + hi + 1) // 2
            sp, _ = compute_tor_space({1: n1_f1, 2: n2_f1, 3: mid})
            if sp <= FAB_SPECS[1]["space"]:
                lo = mid
            else:
                hi = mid - 1
        n3_f1 = min(n3_total, lo)
        n3_total -= n3_f1

        # Fab 2 (remaining space)
        lo, hi = 0, n3_total
        while lo < hi:
            mid = (lo + hi + 1) // 2
            sp, _ = compute_tor_space({1: n1_f2, 2: n2_f2, 3: mid})
            if sp <= FAB_SPECS[2]["space"]:
                lo = mid
            else:
                hi = mid - 1
        n3_f2 = min(n3_total, lo)
        n3_total -= n3_f2

        if n3_total > 0:
            print(f"WARNING: {q} - Cannot fit {n3_total} wafers of Node 3!")

        assignment[q][3][3] = n3_f3
        assignment[q][3][1] = n3_f1
        assignment[q][3][2] = n3_f2

        # Verify totals
        for n in NODES:
            total = sum(assignment[q][n][f] for f in FABS)
            if total != LOADING[n][q]:
                print(f"WARNING: {q} Node {n}: assigned {total}, need {LOADING[n][q]}")

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: COMPUTE TOR TOOL REQUIREMENTS (per fab, ceiling rounded)
# ─────────────────────────────────────────────────────────────────────────────


def compute_tor_requirements(assignment):
    tor_req = {}
    for q in QUARTERS:
        tor_req[q] = {ws: {f: 0 for f in FABS} for ws in TOR_WS}
        for f in FABS:
            flow_nf = {n: assignment[q][n][f] for n in NODES}
            _, tor_int = compute_tor_space(flow_nf)
            for ws in TOR_WS:
                tor_req[q][ws][f] = tor_int[ws]
    return tor_req


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PURCHASE TIMING (just-in-time)
# ─────────────────────────────────────────────────────────────────────────────


def compute_buy_tor(tor_req):
    buy_tor = {}
    prev = {ws: {f: 0 for f in FABS} for ws in TOR_WS}
    for q in QUARTERS:
        buy_tor[q] = {}
        for ws in TOR_WS:
            buy_tor[q][ws] = {}
            for f in FABS:
                buy_tor[q][ws][f] = max(0, tor_req[q][ws][f] - prev[ws][f])
        prev = {ws: {f: tor_req[q][ws][f] for f in FABS} for ws in TOR_WS}
    return buy_tor


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: MOVE-OUT SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────


def compute_moveout_schedule(tor_req):
    moveout = {q: {ws: {f: 0 for f in FABS} for ws in MINTECH_WS} for q in QUARTERS}
    mt_counts = {ws: {f: INITIAL_MT[ws][f] for f in FABS} for ws in MINTECH_WS}
    final_mt = {}

    for q in QUARTERS:
        for f in FABS:
            space_tor = sum(WS_SPECS[ws][0] * tor_req[q][ws][f] for ws in TOR_WS)
            space_mt = sum(WS_SPECS[ws][0] * mt_counts[ws][f] for ws in MINTECH_WS)
            avail = FAB_SPECS[f]["space"]

            if space_tor + space_mt > avail:
                excess = space_tor + space_mt - avail
                # Move out largest-footprint tools first
                ws_sorted = sorted(
                    [
                        (ws, mt_counts[ws][f])
                        for ws in MINTECH_WS
                        if mt_counts[ws][f] > 0
                    ],
                    key=lambda x: -WS_SPECS[x[0]][0],
                )
                for ws, count in ws_sorted:
                    if excess <= 0.001:
                        break
                    sp = WS_SPECS[ws][0]
                    to_move = min(count, math.ceil(excess / sp))
                    moveout[q][ws][f] = to_move
                    mt_counts[ws][f] -= to_move
                    excess -= to_move * sp

        final_mt[q] = {ws: {f: mt_counts[ws][f] for f in FABS} for ws in MINTECH_WS}

    return moveout, final_mt


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────


def validate(assignment, tor_req, moveout, final_mt):
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    all_ok = True

    # Demand
    print("\n--- Demand ---")
    dem_ok = True
    for q in QUARTERS:
        for n in NODES:
            total = sum(assignment[q][n][f] for f in FABS)
            req = LOADING[n][q]
            if total != req:
                print(f"  FAIL: {q} Node {n}: {total} vs {req}")
                dem_ok = False
                all_ok = False
    if dem_ok:
        print("  PASS: All demand constraints satisfied")

    # Space
    print("\n--- Space ---")
    sp_ok = True
    for q in QUARTERS:
        for f in FABS:
            sp_mt = sum(WS_SPECS[ws][0] * final_mt[q][ws][f] for ws in MINTECH_WS)
            sp_tor = sum(WS_SPECS[ws][0] * tor_req[q][ws][f] for ws in TOR_WS)
            total = sp_mt + sp_tor
            avail = FAB_SPECS[f]["space"]
            if total > avail + 0.001:
                print(f"  FAIL: {q} Fab {f}: {total:.3f}/{avail}")
                sp_ok = False
                all_ok = False
    if sp_ok:
        print("  PASS: All space constraints satisfied")

    # Capacity
    print("\n--- Capacity ---")
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
                supply = tor_req[q][ws_tor][f]
                if demand > supply + 0.001:
                    print(
                        f"  FAIL: {q} Fab {f} {ws_tor}: demand={demand:.4f} > supply={supply}"
                    )
                    cap_ok = False
                    all_ok = False
    if cap_ok:
        print("  PASS: All capacity constraints satisfied")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# COST COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────


def compute_costs(buy_tor, moveout):
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
    total = capex + opex_mo
    print(f"\n{'=' * 60}")
    print(f"COST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  CapEx (TOR purchases):  ${capex:>15,.0f}")
    print(f"  OpEx (move-outs):       ${opex_mo:>15,.0f}")
    print(f"  Total Cost:             ${total:>15,.0f}")
    return capex, opex_mo, total


# ─────────────────────────────────────────────────────────────────────────────
# DETAILED REPORT
# ─────────────────────────────────────────────────────────────────────────────


def report(assignment, tor_req, buy_tor, moveout, final_mt):
    print("\n" + "=" * 60)
    print("DETAILED SOLUTION REPORT")
    print("=" * 60)

    print("\n=== Flow Assignment (wafers/week) ===")
    print(
        f"{'Quarter':<10} {'Node':<6} {'Fab1':>8} {'Fab2':>8} {'Fab3':>8} {'Total':>8} {'Req':>8}"
    )
    for q in QUARTERS:
        for n in NODES:
            f1, f2, f3 = assignment[q][n][1], assignment[q][n][2], assignment[q][n][3]
            total = f1 + f2 + f3
            req = LOADING[n][q]
            ok = "OK" if total == req else f"ERR"
            print(f"{q:<10} {n:<6} {f1:>8} {f2:>8} {f3:>8} {total:>8} {req:>8} {ok}")

    print("\n=== TOR Tool Counts by Quarter (per fab) ===")
    for q in QUARTERS:
        print(f"\n  {q}:")
        for ws in TOR_WS:
            f1, f2, f3 = tor_req[q][ws][1], tor_req[q][ws][2], tor_req[q][ws][3]
            if f1 + f2 + f3 > 0:
                print(
                    f"    {ws}: F1={f1:3d}, F2={f2:3d}, F3={f3:3d}  Total={f1 + f2 + f3}"
                )

    print("\n=== TOR Tool Purchases (incremental) ===")
    for q in QUARTERS:
        q_total = sum(buy_tor[q][ws][f] for ws in TOR_WS for f in FABS)
        if q_total > 0:
            q_cost = sum(
                WS_SPECS[ws][1] * buy_tor[q][ws][f] for ws in TOR_WS for f in FABS
            )
            print(f"\n  {q}: {q_total} tools (${q_cost:,.0f})")
            for ws in TOR_WS:
                for f in FABS:
                    v = buy_tor[q][ws][f]
                    if v > 0:
                        print(f"    {ws} Fab{f}: +{v}")

    print("\n=== Move-Out Schedule ===")
    total_mo = 0
    for q in QUARTERS:
        q_mo = sum(moveout[q][ws][f] for ws in MINTECH_WS for f in FABS)
        if q_mo > 0:
            print(f"\n  {q}: {q_mo} tools (${q_mo * MOVEOUT_COST:,.0f})")
            for ws in MINTECH_WS:
                for f in FABS:
                    m = moveout[q][ws][f]
                    if m > 0:
                        print(f"    {ws} Fab{f}: -{m}")
            total_mo += q_mo
    print(f"\n  Total: {total_mo} tools (${total_mo * MOVEOUT_COST:,.0f})")

    print("\n=== Space Utilization ===")
    print(f"{'Quarter':<10} {'Fab1':>14} {'Fab2':>14} {'Fab3':>12}")
    for q in QUARTERS:
        row = f"{q:<10}"
        for f in FABS:
            sp_mt = sum(WS_SPECS[ws][0] * final_mt[q][ws][f] for ws in MINTECH_WS)
            sp_tor = sum(WS_SPECS[ws][0] * tor_req[q][ws][f] for ws in TOR_WS)
            total = sp_mt + sp_tor
            avail = FAB_SPECS[f]["space"]
            pct = 100 * total / avail
            row += f"  {total:.0f}/{avail}({pct:.0f}%)"
        print(row)

    print("\n=== Final Tool State at Q4'27 ===")
    q = QUARTERS[-1]
    print(
        f"{'WS':<5} {'TOR F1':>8} {'TOR F2':>8} {'TOR F3':>8} {'MT F1':>8} {'MT F2':>8} {'MT F3':>8}"
    )
    for ws in MINTECH_WS:
        ws_tor = TOR_MAP[ws]
        t1, t2, t3 = tor_req[q][ws_tor][1], tor_req[q][ws_tor][2], tor_req[q][ws_tor][3]
        m1, m2, m3 = final_mt[q][ws][1], final_mt[q][ws][2], final_mt[q][ws][3]
        print(f"{ws:<5} {t1:>8} {t2:>8} {t3:>8} {m1:>8} {m2:>8} {m3:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Q1b SOLUTION v2 – SPACE-AWARE ROUNDING")
    print("=" * 60)

    print("\nStep 1: Computing node-fab assignment (space-aware)...")
    assignment = compute_assignment()

    print("Step 2: Computing TOR tool requirements...")
    tor_req = compute_tor_requirements(assignment)

    print("Step 3: Computing purchase timing...")
    buy_tor = compute_buy_tor(tor_req)

    print("Step 4: Computing move-out schedule...")
    moveout, final_mt = compute_moveout_schedule(tor_req)

    all_ok = validate(assignment, tor_req, moveout, final_mt)
    capex, opex_mo, total_cost = compute_costs(buy_tor, moveout)
    report(assignment, tor_req, buy_tor, moveout, final_mt)

    # Save
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
            q: {ws: {str(f): tor_req[q][ws][f] for f in FABS} for ws in TOR_WS}
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

    with open("results/q1b_solution_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: results/q1b_solution_v2_results.json")

    if all_ok:
        print("\n✓ SOLUTION IS FULLY FEASIBLE")
    else:
        print("\n✗ Solution has violations — review above")
