"""
Script 5: Q1b Definitive LP Solver
====================================
Correct formulation:
- All production uses TOR tools (optimal strategy proven by space analysis)
- Mintech tools are moved out progressively to free space for TOR tools
- The LP optimizes: (1) node-fab assignment, (2) TOR tool purchases, (3) move-out schedule
- Objective: minimize CapEx (TOR) + OpEx (move-outs)

Key insight: Total space needed at Q4'27 = 3,283 m² (all-TOR) < 3,500 m² (total available).
This means the problem IS feasible if we move out enough mintech tools.
The initial mintech tools use 2,869 m², so we need to move out nearly all of them
to make room for TOR tools.
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

# Initial tool counts
INITIAL_MT = {ws: {f: FAB_SPECS[f]["tools"][ws] for f in FABS} for ws in MINTECH_WS}
TOTAL_INITIAL_TOOLS = sum(INITIAL_MT[ws][f] for ws in MINTECH_WS for f in FABS)
print(f"Total initial mintech tools: {TOTAL_INITIAL_TOOLS}")
print(
    f"Initial space used: {sum(WS_SPECS[ws][0] * INITIAL_MT[ws][f] for ws in MINTECH_WS for f in FABS):.1f} m²"
)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD LP MODEL
# ─────────────────────────────────────────────────────────────────────────────


def build_model():
    model = pulp.LpProblem("Q1b_Final_LP", pulp.LpMinimize)

    # ── Variables ───────────────────────────────────────────────────────────

    # flow[q][n][f]: wafers of node n processed in fab f in quarter q
    flow = {
        q: {
            n: {f: pulp.LpVariable(f"w_{q}_{n}_{f}", lowBound=0) for f in FABS}
            for n in NODES
        }
        for q in QUARTERS
    }

    # tor[q][ws][f]: TOR tools of type ws in fab f at start of quarter q
    tor = {
        q: {
            ws: {f: pulp.LpVariable(f"t_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in TOR_WS
        }
        for q in QUARTERS
    }

    # mt[q][ws][f]: mintech tools remaining in fab f at start of quarter q
    mt = {
        q: {
            ws: {f: pulp.LpVariable(f"m_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in MINTECH_WS
        }
        for q in QUARTERS
    }

    # buy_tor[q][ws][f]: TOR tools purchased in quarter q
    buy_tor = {
        q: {
            ws: {f: pulp.LpVariable(f"b_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in TOR_WS
        }
        for q in QUARTERS
    }

    # moveout[q][ws][f]: mintech tools moved out in quarter q
    moveout = {
        q: {
            ws: {f: pulp.LpVariable(f"o_{q}_{ws}_{f}", lowBound=0) for f in FABS}
            for ws in MINTECH_WS
        }
        for q in QUARTERS
    }

    # ── Link constraints ────────────────────────────────────────────────────

    for qi, q in enumerate(QUARTERS):
        q_prev = QUARTERS[qi - 1] if qi > 0 else None

        for ws in MINTECH_WS:
            for f in FABS:
                init = INITIAL_MT[ws][f]
                if qi == 0:
                    model += (
                        mt[q][ws][f] == init - moveout[q][ws][f],
                        f"mt_link_{q}_{ws}_{f}",
                    )
                else:
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
    opex_mo = pulp.lpSum(
        MOVEOUT_COST * moveout[q][ws][f]
        for q in QUARTERS
        for ws in MINTECH_WS
        for f in FABS
    )

    model += capex + opex_mo, "Total_Cost"

    # ── Constraints ─────────────────────────────────────────────────────────

    # C1. Demand
    for q in QUARTERS:
        for n in NODES:
            model += (
                pulp.lpSum(flow[q][n][f] for f in FABS) == LOADING[n][q],
                f"demand_{q}_{n}",
            )

    # C2. TOR capacity: for each (q, ws_tor, f)
    for q in QUARTERS:
        for ws_tor in TOR_WS:
            _, _, util = WS_SPECS[ws_tor]
            for f in FABS:
                usage = []
                for n in NODES:
                    for _, ws_mt, rpt_mt, ws_t, rpt_t in PROCESS_STEPS[n]:
                        if ws_t == ws_tor:
                            usage.append(flow[q][n][f] * rpt_t / (MIN_PER_WEEK * util))
                if usage:
                    model += (
                        pulp.lpSum(usage) <= tor[q][ws_tor][f],
                        f"cap_{q}_{ws_tor}_{f}",
                    )

    # C3. Space: mt tools + tor tools <= fab space
    for q in QUARTERS:
        for f in FABS:
            sp_mt = pulp.lpSum(WS_SPECS[ws][0] * mt[q][ws][f] for ws in MINTECH_WS)
            sp_tor = pulp.lpSum(WS_SPECS[ws][0] * tor[q][ws][f] for ws in TOR_WS)
            model += (sp_mt + sp_tor <= FAB_SPECS[f]["space"], f"space_{q}_{f}")

    # C4. Mintech tools >= 0 (enforced by lowBound)
    # C5. Move-outs >= 0 (enforced by lowBound)
    # C6. TOR tools non-decreasing (enforced by buy_tor >= 0)

    return model, flow, tor, mt, buy_tor, moveout


# ─────────────────────────────────────────────────────────────────────────────
# SOLVE
# ─────────────────────────────────────────────────────────────────────────────


def solve(model, flow, tor, mt, buy_tor, moveout):
    n_v = len(model.variables())
    n_c = len(model.constraints)
    print(f"\nModel: {n_v} variables, {n_c} constraints.")
    print("Solving LP...")

    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=120)
    status = model.solve(solver)
    elapsed = time.time() - t0

    print(f"Status: {pulp.LpStatus[status]}, Time: {elapsed:.2f}s")

    if pulp.value(model.objective) is None:
        print("No solution found.")
        return None

    obj = pulp.value(model.objective)
    print(f"LP Objective: ${obj:,.0f}")

    # Extract
    R = {
        "status": pulp.LpStatus[status],
        "lp_obj": obj,
        "flow": {},
        "tor": {},
        "mt": {},
        "buy_tor": {},
        "moveout": {},
    }

    for q in QUARTERS:
        R["flow"][q] = {
            n: {f: max(0, pulp.value(flow[q][n][f]) or 0) for f in FABS} for n in NODES
        }
        R["tor"][q] = {
            ws: {f: max(0, pulp.value(tor[q][ws][f]) or 0) for f in FABS}
            for ws in TOR_WS
        }
        R["mt"][q] = {
            ws: {f: max(0, pulp.value(mt[q][ws][f]) or 0) for f in FABS}
            for ws in MINTECH_WS
        }
        R["buy_tor"][q] = {
            ws: {f: max(0, pulp.value(buy_tor[q][ws][f]) or 0) for f in FABS}
            for ws in TOR_WS
        }
        R["moveout"][q] = {
            ws: {f: max(0, pulp.value(moveout[q][ws][f]) or 0) for f in FABS}
            for ws in MINTECH_WS
        }

    return R


# ─────────────────────────────────────────────────────────────────────────────
# ROUND AND VALIDATE
# ─────────────────────────────────────────────────────────────────────────────


def round_and_validate(R):
    """Round LP solution to integers and validate all constraints."""
    print("\n=== Rounding to integers ===")

    F = {"flow": {}, "tor": {}, "mt": {}, "buy_tor": {}, "moveout": {}}

    # Round flow to nearest integer, ensuring demand is met
    for q in QUARTERS:
        F["flow"][q] = {}
        for n in NODES:
            raw = {f: R["flow"][q][n][f] for f in FABS}
            total_raw = sum(raw.values())
            req = LOADING[n][q]
            # Scale to meet demand exactly
            if total_raw > 0:
                scale = req / total_raw
                scaled = {f: raw[f] * scale for f in FABS}
            else:
                scaled = {f: req / 3 for f in FABS}
            # Round: floor all, then add remainder to largest
            floored = {f: math.floor(scaled[f]) for f in FABS}
            remainder = req - sum(floored.values())
            # Add remainder to fab with largest fractional part
            fracs = sorted(FABS, key=lambda f: -(scaled[f] - floored[f]))
            for i in range(int(round(remainder))):
                floored[fracs[i % 3]] += 1
            F["flow"][q][n] = floored

    # Round up TOR tools (ceiling to ensure capacity)
    for q in QUARTERS:
        F["tor"][q] = {}
        for ws in TOR_WS:
            F["tor"][q][ws] = {}
            _, _, util = WS_SPECS[ws]
            for f in FABS:
                # Compute actual demand with rounded flow
                demand = 0
                for n in NODES:
                    wafers = F["flow"][q][n][f]
                    for _, ws_mt, rpt_mt, ws_t, rpt_t in PROCESS_STEPS[n]:
                        if ws_t == ws:
                            demand += wafers * rpt_t / (MIN_PER_WEEK * util)
                F["tor"][q][ws][f] = math.ceil(demand)

    # Compute move-outs needed to satisfy space constraints
    F["mt"] = {}
    F["moveout"] = {}
    current_mt = {ws: {f: INITIAL_MT[ws][f] for f in FABS} for ws in MINTECH_WS}

    for q in QUARTERS:
        F["mt"][q] = {}
        F["moveout"][q] = {}

        # Initialize with current state
        for ws in MINTECH_WS:
            F["mt"][q][ws] = {f: current_mt[ws][f] for f in FABS}
            F["moveout"][q][ws] = {f: 0 for f in FABS}

        # Check space for each fab and move out as needed
        for f in FABS:
            space_tor = sum(WS_SPECS[ws][0] * F["tor"][q][ws][f] for ws in TOR_WS)
            space_mt = sum(WS_SPECS[ws][0] * current_mt[ws][f] for ws in MINTECH_WS)
            avail = FAB_SPECS[f]["space"]

            if space_tor + space_mt > avail:
                excess = space_tor + space_mt - avail
                # Move out mintech tools: prioritize by space freed per dollar
                # (move out tools with largest footprint first)
                ws_sorted = sorted(
                    [
                        (ws, current_mt[ws][f])
                        for ws in MINTECH_WS
                        if current_mt[ws][f] > 0
                    ],
                    key=lambda x: -WS_SPECS[x[0]][0],  # largest space first
                )
                for ws, count in ws_sorted:
                    if excess <= 0:
                        break
                    sp = WS_SPECS[ws][0]
                    to_move = min(count, math.ceil(excess / sp))
                    F["moveout"][q][ws][f] = to_move
                    F["mt"][q][ws][f] = count - to_move
                    excess -= to_move * sp

        # Update current_mt
        for ws in MINTECH_WS:
            for f in FABS:
                current_mt[ws][f] = F["mt"][q][ws][f]

    # Compute buy_tor (incremental)
    F["buy_tor"] = {}
    prev_tor = {ws: {f: 0 for f in FABS} for ws in TOR_WS}
    for q in QUARTERS:
        F["buy_tor"][q] = {}
        for ws in TOR_WS:
            F["buy_tor"][q][ws] = {}
            for f in FABS:
                bought = F["tor"][q][ws][f] - prev_tor[ws][f]
                F["buy_tor"][q][ws][f] = max(0, bought)
        prev_tor = {ws: {f: F["tor"][q][ws][f] for f in FABS} for ws in TOR_WS}

    # ── Validation ──────────────────────────────────────────────────────────

    print("\n=== Demand Validation ===")
    dem_ok = True
    for q in QUARTERS:
        for n in NODES:
            total = sum(F["flow"][q][n][f] for f in FABS)
            req = LOADING[n][q]
            if abs(total - req) > 0:
                print(f"  VIOLATED: {q} Node {n}: got {total}, need {req}")
                dem_ok = False
    if dem_ok:
        print("  All demand constraints satisfied!")

    print("\n=== Space Validation ===")
    space_ok = True
    for q in QUARTERS:
        for f in FABS:
            sp_mt = sum(WS_SPECS[ws][0] * F["mt"][q][ws][f] for ws in MINTECH_WS)
            sp_tor = sum(WS_SPECS[ws][0] * F["tor"][q][ws][f] for ws in TOR_WS)
            total = sp_mt + sp_tor
            avail = FAB_SPECS[f]["space"]
            if total > avail + 0.01:
                print(f"  VIOLATED: {q} Fab {f}: {total:.2f}/{avail}")
                space_ok = False
    if space_ok:
        print("  All space constraints satisfied!")

    print("\n=== Capacity Validation ===")
    cap_ok = True
    for q in QUARTERS:
        for ws_tor in TOR_WS:
            _, _, util = WS_SPECS[ws_tor]
            for f in FABS:
                demand = 0
                for n in NODES:
                    wafers = F["flow"][q][n][f]
                    for _, ws_mt, rpt_mt, ws_t, rpt_t in PROCESS_STEPS[n]:
                        if ws_t == ws_tor:
                            demand += wafers * rpt_t / (MIN_PER_WEEK * util)
                supply = F["tor"][q][ws_tor][f]
                if demand > supply + 0.01:
                    print(
                        f"  VIOLATED: {q} Fab {f} {ws_tor}: demand={demand:.3f} > supply={supply}"
                    )
                    cap_ok = False
    if cap_ok:
        print("  All capacity constraints satisfied!")

    # ── Cost Summary ────────────────────────────────────────────────────────

    capex = sum(
        WS_SPECS[ws][1] * F["buy_tor"][q][ws][f]
        for q in QUARTERS
        for ws in TOR_WS
        for f in FABS
    )
    opex_mo = sum(
        MOVEOUT_COST * F["moveout"][q][ws][f]
        for q in QUARTERS
        for ws in MINTECH_WS
        for f in FABS
    )
    F["capex"] = capex
    F["opex_moveout"] = opex_mo
    F["total_cost"] = capex + opex_mo

    print(f"\n  CapEx (TOR purchases):  ${capex:>15,.0f}")
    print(f"  OpEx (move-outs):       ${opex_mo:>15,.0f}")
    print(f"  Total Cost:             ${capex + opex_mo:>15,.0f}")

    # ── Detailed Reports ────────────────────────────────────────────────────

    print("\n=== Flow Assignment (wafers/week) ===")
    print(
        f"{'Quarter':<10} {'Node':<6} {'Fab1':>8} {'Fab2':>8} {'Fab3':>8} {'Total':>8} {'Req':>8}"
    )
    for q in QUARTERS:
        for n in NODES:
            f1, f2, f3 = F["flow"][q][n][1], F["flow"][q][n][2], F["flow"][q][n][3]
            total = f1 + f2 + f3
            req = LOADING[n][q]
            ok = "OK" if total == req else f"ERR({req})"
            print(f"{q:<10} {n:<6} {f1:>8} {f2:>8} {f3:>8} {total:>8} {req:>8} {ok}")

    print("\n=== TOR Tool Purchases by Quarter ===")
    print(f"{'Quarter':<10}", end="")
    for ws in TOR_WS:
        for f in FABS:
            print(f"  {ws}/F{f}", end="")
    print()
    for q in QUARTERS:
        print(f"{q:<10}", end="")
        for ws in TOR_WS:
            for f in FABS:
                v = F["buy_tor"][q][ws][f]
                print(f"  {v:>6}", end="")
        print()

    print("\n=== Move-Out Schedule ===")
    total_mo = 0
    for q in QUARTERS:
        q_mo = sum(F["moveout"][q][ws][f] for ws in MINTECH_WS for f in FABS)
        if q_mo > 0:
            print(f"  {q}: {q_mo} tools (${q_mo * MOVEOUT_COST:,.0f})")
            for ws in MINTECH_WS:
                for f in FABS:
                    m = F["moveout"][q][ws][f]
                    if m > 0:
                        print(f"    {ws} Fab{f}: -{m}")
            total_mo += q_mo
    print(f"  Total: {total_mo} tools (${total_mo * MOVEOUT_COST:,.0f})")

    print("\n=== Final Tool Counts at Q4'27 ===")
    q = QUARTERS[-1]
    for ws in TOR_WS:
        counts = [F["tor"][q][ws][f] for f in FABS]
        if any(c > 0 for c in counts):
            print(
                f"  {ws:3s}: F1={counts[0]:4d}, F2={counts[1]:4d}, F3={counts[2]:4d}  Total={sum(counts)}"
            )
    print()
    for ws in MINTECH_WS:
        counts = [F["mt"][q][ws][f] for f in FABS]
        if any(c > 0 for c in counts):
            print(
                f"  {ws:3s}: F1={counts[0]:4d}, F2={counts[1]:4d}, F3={counts[2]:4d}  Total={sum(counts)}"
            )

    return F


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Q1b DEFINITIVE LP SOLVER")
    print("=" * 60)

    model, flow, tor, mt, buy_tor, moveout = build_model()
    lp_results = solve(model, flow, tor, mt, buy_tor, moveout)

    if lp_results:
        final = round_and_validate(lp_results)

        def convert(obj):
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, float):
                return round(obj, 6)
            return obj

        with open("results/q1b_final_results.json", "w") as f:
            json.dump(convert(final), f, indent=2)
        print("\nFinal results saved to: results/q1b_final_results.json")
    else:
        print("LP failed — check model formulation.")
