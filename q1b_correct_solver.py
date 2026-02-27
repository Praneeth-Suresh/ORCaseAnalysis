"""
Q1b Correct LP Solver
======================
NUS-ISE BACC 2026 — Micron Technology Chip Production Optimization

KEY INSIGHTS:
1. Q1'26 tools are FIXED (cannot change). Only mintech tools exist in Q1'26.
2. Loading is PER STEP PER FAB — different steps can have different fab assignments.
3. Tool requirement = SUM (not max) of loading*RPT/(10080*util) across all steps using that WS.
4. Cross-fab transfers occur when loading changes between consecutive steps.
5. Transfer cost = $650 per wafer per transfer event (= $50 * 13 weeks).
6. From Q2'26 onwards: can purchase TOR tools (CapEx) or move out mintech tools ($1M each).

OBJECTIVE: Minimize total CapEx + MoveOut + Transfer cost over all 8 quarters.

DATA SOURCE: All problem parameters are read from the Excel answer sheet
             (sheet: 'Input Data + Q2') at runtime.
"""

import json
import re
import time as time_module
from pathlib import Path

import openpyxl
from pulp import *

# ============================================================
# EXCEL DATA LOADER
# ============================================================

# Locate the Excel file relative to this script
_ROOT = Path(__file__).resolve().parent
EXCEL_PATH = _ROOT / "5)BACC2026ParticipantAnswerSheet.xlsx"


def _parse_capex(cell_value) -> float:
    """Parse a CAPEX string like '4.5M ' or '6.0M ' into a float (e.g. 4500000.0)."""
    if isinstance(cell_value, (int, float)):
        return float(cell_value)
    text = str(cell_value).strip().upper().replace(",", "")
    match = re.match(r"([\d.]+)\s*M", text)
    if match:
        return float(match.group(1)) * 1_000_000
    return float(text)


def _parse_tool_cell(cell_value) -> int:
    """Parse a fab tool-count cell like 'A: 50' or 'F:  90' into the integer count."""
    if isinstance(cell_value, (int, float)):
        return int(cell_value)
    # Format is  'WS: count'  (possibly extra spaces)
    parts = str(cell_value).split(":")
    return int(parts[1].strip())


def load_data_from_excel(excel_path: Path = EXCEL_PATH) -> dict:
    """
    Read all problem parameters from the 'Input Data + Q2' sheet of the Excel file.

    Returns a dict with keys:
        quarters        – list of 8 quarter strings
        demand          – {node: [q0, q1, ..., q7]}
        steps           – {node: [(ws_mt, rpt_mt, ws_tor, rpt_tor), ...]}
        space           – {ws: float}
        capex           – {ws: float}
        util            – {ws: float}
        q1_tools        – {ws: [fab1_count, fab2_count, fab3_count]}
        fab_space       – {1: int, 2: int, 3: int}
        moveout_cost    – int
        xfer_cost       – int  ($50 * 13 weeks)
        min_per_week    – int  (7 * 24 * 60)
    """
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb["Input Data + Q2"]

    def v(row, col):
        """Return the value at (row, col); col is 1-based."""
        return ws.cell(row, col).value

    # ── Quarters (row 3, cols 3–10) ──────────────────────────────────────
    quarters = []
    for col in range(3, 11):
        raw = str(v(3, col)).strip().replace("\xa0", "")
        quarters.append(raw)

    # ── Demand (rows 4–6, cols 3–10) ────────────────────────────────────
    demand = {}
    for node_idx, row in enumerate([4, 5, 6], start=1):
        demand[node_idx] = [int(v(row, col)) for col in range(3, 11)]

    # ── Process steps ───────────────────────────────────────────────────
    # Node 1 cols: step=3, ws=4, rpt=5, ws_tor=6, rpt_tor=7  (rows 10–20)
    # Node 2 cols: step=10, ws=11, rpt=12, ws_tor=13, rpt_tor=14 (rows 10–24)
    # Node 3 cols: step=17, ws=18, rpt=19, ws_tor=20, rpt_tor=21 (rows 10–26)
    node_col_offsets = {
        1: (3, 4, 5, 6, 7),  # (name_col, ws_col, rpt_col, ws_tor_col, rpt_tor_col)
        2: (10, 11, 12, 13, 14),
        3: (17, 18, 19, 20, 21),
    }
    steps = {}
    for node, (nc, wc, rc, wtc, rtc) in node_col_offsets.items():
        node_steps = []
        for row in range(10, 40):  # generous upper bound
            ws_val = v(row, wc)
            rpt_val = v(row, rc)
            ws_tor_val = v(row, wtc)
            rpt_tor_val = v(row, rtc)
            if ws_val is None or rpt_val is None:
                break
            node_steps.append(
                (
                    str(ws_val).strip(),
                    int(rpt_val),
                    str(ws_tor_val).strip(),
                    int(rpt_tor_val),
                )
            )
        steps[node] = node_steps

    # ── WS specs – mintech (cols 25–30) ─────────────────────────────────
    # Row 9:  headers  A B C D E F
    # Row 14: util
    # Row 15: capex (text "4.5M")
    # Row 16: space (m²)
    mt_ws_names = [str(v(9, c)).strip() for c in range(25, 31)]  # A..F
    util_mt = {mt_ws_names[i]: float(v(14, 25 + i)) for i in range(6)}
    capex_mt = {mt_ws_names[i]: _parse_capex(v(15, 25 + i)) for i in range(6)}
    space_mt = {mt_ws_names[i]: float(v(16, 25 + i)) for i in range(6)}

    # ── WS specs – TOR (cols 25–30) ─────────────────────────────────────
    # Row 18: headers  A+ B+ C+ D+ E+ F+
    # Row 23: util
    # Row 24: capex
    # Row 25: space
    tor_ws_names = [str(v(18, c)).strip() for c in range(25, 31)]  # A+..F+
    util_tor = {tor_ws_names[i]: float(v(23, 25 + i)) for i in range(6)}
    capex_tor = {tor_ws_names[i]: _parse_capex(v(24, 25 + i)) for i in range(6)}
    space_tor = {tor_ws_names[i]: float(v(25, 25 + i)) for i in range(6)}

    space = {**space_mt, **space_tor}
    capex = {**capex_mt, **capex_tor}
    util = {**util_mt, **util_tor}

    # ── Fab total space (row 29, cols 25–27) ────────────────────────────
    fab_space = {
        1: int(v(29, 25)),
        2: int(v(29, 26)),
        3: int(v(29, 27)),
    }

    # ── Initial tool counts per fab (rows 30–35, cols 25–27) ────────────
    # Each row corresponds to one mintech WS (A, B, C, D, E, F)
    q1_tools = {}
    for row in range(30, 36):
        fab1_val = v(row, 25)
        if fab1_val is None:
            continue
        count_f1 = _parse_tool_cell(fab1_val)
        count_f2 = _parse_tool_cell(v(row, 26))
        count_f3 = _parse_tool_cell(v(row, 27))
        # Determine the WS letter from the cell text (e.g. "A: 50" → "A")
        ws_letter = str(fab1_val).split(":")[0].strip()
        q1_tools[ws_letter] = [count_f1, count_f2, count_f3]

    # TOR tools always start at 0 (read from row 22 but will always be 0)
    for ws_name in tor_ws_names:
        q1_tools[ws_name] = [0, 0, 0]

    wb.close()

    return {
        "quarters": quarters,
        "demand": demand,
        "steps": steps,
        "space": space,
        "capex": capex,
        "util": util,
        "q1_tools": q1_tools,
        "fab_space": fab_space,
        "moveout_cost": 1_000_000,
        "xfer_cost": 50 * 13,  # $650 per wafer per cross-fab transfer
        "min_per_week": 7 * 24 * 60,
    }


# ============================================================
# LOAD ALL DATA FROM EXCEL
# ============================================================
print(f"Loading data from: {EXCEL_PATH.name}")
_DATA = load_data_from_excel()

MIN_PER_WEEK = _DATA["min_per_week"]
QUARTERS = _DATA["quarters"]
WS_MT = ["A", "B", "C", "D", "E", "F"]
WS_TOR = ["A+", "B+", "C+", "D+", "E+", "F+"]
WS_ALL = WS_MT + WS_TOR
FABS = [1, 2, 3]
NODES = [1, 2, 3]
SPACE = _DATA["space"]
CAPEX = _DATA["capex"]
UTIL = _DATA["util"]
MOVEOUT_COST = _DATA["moveout_cost"]
XFER_COST = _DATA["xfer_cost"]
FAB_SPACE = _DATA["fab_space"]
Q1_TOOLS = _DATA["q1_tools"]
DEMAND = _DATA["demand"]
# STEPS: convert list-of-tuples loaded from Excel (same shape as original)
STEPS = _DATA["steps"]

# ============================================================
# HELPER: which (WS, RPT) pairs does each step use?
# In Q1'26: only mintech. From Q2'26: can use either mintech or TOR.
# We model this by having separate loading variables for mintech and TOR.
# ============================================================


def solve_quarter(q_idx, prev_tools, verbose=True):
    """Solve one quarter. Returns (L_sol, T_sol, cost)."""
    quarter = QUARTERS[q_idx]
    D = {n: DEMAND[n][q_idx] for n in NODES}
    is_q1 = q_idx == 0

    prob = LpProblem(f"q{q_idx}", LpMinimize)

    # Loading variables: L_mt[n,s,f] for mintech, L_tor[n,s,f] for TOR
    # In Q1'26, L_tor = 0 (no TOR tools available)
    # Loading is integer so assigned loading always matches tool-feasible loading.
    L_mt = {}
    L_tor = {}
    for node in NODES:
        for s_idx in range(len(STEPS[node])):
            for fab in FABS:
                L_mt[(node, s_idx, fab)] = LpVariable(
                    f"Lmt_{node}_{s_idx}_{fab}", lowBound=0, cat="Integer"
                )
                if not is_q1:
                    L_tor[(node, s_idx, fab)] = LpVariable(
                        f"Ltor_{node}_{s_idx}_{fab}", lowBound=0, cat="Integer"
                    )

    # Total loading per (node, step, fab)
    def L_total(node, s_idx, fab):
        if is_q1:
            return L_mt[(node, s_idx, fab)]
        return L_mt[(node, s_idx, fab)] + L_tor[(node, s_idx, fab)]

    # Tool count variables (Q2'26 onwards)
    if not is_q1:
        T = {
            ws: [LpVariable(f"T_{ws}_{f}", lowBound=0, cat="Integer") for f in range(3)]
            for ws in WS_ALL
        }
        P = {
            ws: [LpVariable(f"P_{ws}_{f}", lowBound=0, cat="Integer") for f in range(3)]
            for ws in WS_ALL
        }
        MO = {
            ws: [
                LpVariable(f"MO_{ws}_{f}", lowBound=0, cat="Integer") for f in range(3)
            ]
            for ws in WS_ALL
        }

    # Transfer delta variables
    dm = {}  # delta_minus: wafers leaving fab f between step s and s+1
    dp = {}  # delta_plus:  wafers entering fab f between step s and s+1
    for node in NODES:
        for s_idx in range(len(STEPS[node]) - 1):
            for fab in FABS:
                dm[(node, s_idx, fab)] = LpVariable(
                    f"dm_{node}_{s_idx}_{fab}", lowBound=0
                )
                dp[(node, s_idx, fab)] = LpVariable(
                    f"dp_{node}_{s_idx}_{fab}", lowBound=0
                )

    # ---- CONSTRAINTS ----

    # 1. Demand: sum_f L_total[n,s,f] = D[n]
    for node in NODES:
        for s_idx in range(len(STEPS[node])):
            prob += (
                lpSum(L_total(node, s_idx, fab) for fab in FABS) == D[node],
                f"dem_{node}_{s_idx}",
            )

    # 2. Tool capacity
    for ws in WS_ALL:
        is_tor = ws.endswith("+")
        base_ws = ws.rstrip("+")
        for fab_idx, fab in enumerate(FABS):
            # Collect steps that use this WS
            ws_steps = []
            for node in NODES:
                for s_idx, (ws_mt, rpt_mt, ws_t, rpt_t) in enumerate(STEPS[node]):
                    if not is_tor and ws_mt == ws:
                        ws_steps.append((node, s_idx, rpt_mt, L_mt[(node, s_idx, fab)]))
                    elif is_tor and ws_t == ws and not is_q1:
                        ws_steps.append((node, s_idx, rpt_t, L_tor[(node, s_idx, fab)]))

            if not ws_steps:
                continue

            req = lpSum(
                loading * rpt / (MIN_PER_WEEK * UTIL[ws])
                for _, _, rpt, loading in ws_steps
            )

            if is_q1:
                avail = prev_tools[ws][fab_idx]
                prob += req <= avail, f"cap_{ws}_{fab_idx}"
            else:
                prob += req <= T[ws][fab_idx], f"cap_{ws}_{fab_idx}"

    # 3. Tool inventory (Q2'26 onwards)
    if not is_q1:
        for ws in WS_ALL:
            for fab_idx in range(3):
                prev = prev_tools[ws][fab_idx]
                prob += (
                    T[ws][fab_idx] == prev + P[ws][fab_idx] - MO[ws][fab_idx],
                    f"inv_{ws}_{fab_idx}",
                )
                prob += MO[ws][fab_idx] <= prev, f"maxout_{ws}_{fab_idx}"

    # 4. Space (Q2'26 onwards)
    if not is_q1:
        for fab_idx, fab in enumerate(FABS):
            prob += (
                lpSum(T[ws][fab_idx] * SPACE[ws] for ws in WS_ALL) <= FAB_SPACE[fab],
                f"space_{fab_idx}",
            )

    # 5. Transfer delta: dp - dm = L_total[s+1,f] - L_total[s,f]
    for node in NODES:
        for s_idx in range(len(STEPS[node]) - 1):
            for fab in FABS:
                prob += (
                    dp[(node, s_idx, fab)] - dm[(node, s_idx, fab)]
                    == L_total(node, s_idx + 1, fab) - L_total(node, s_idx, fab),
                    f"delta_{node}_{s_idx}_{fab}",
                )

    # ---- OBJECTIVE ----
    xfer_cost = lpSum(
        dm[(node, s_idx, fab)] * XFER_COST
        for node in NODES
        for s_idx in range(len(STEPS[node]) - 1)
        for fab in FABS
    )

    if is_q1:
        prob += xfer_cost
    else:
        capex_cost = lpSum(P[ws][f] * CAPEX[ws] for ws in WS_ALL for f in range(3))
        mo_cost = lpSum(MO[ws][f] * MOVEOUT_COST for ws in WS_ALL for f in range(3))
        prob += capex_cost + mo_cost + xfer_cost

    # ---- SOLVE ----
    solver = PULP_CBC_CMD(msg=0, timeLimit=180, gapRel=0.01)
    prob.solve(solver)

    if verbose:
        print(
            f"  {quarter}: {LpStatus[prob.status]}, Obj=${value(prob.objective) / 1e6:.2f}M"
        )

    if prob.status not in [1, -2]:  # 1=Optimal, -2=Infeasible but has solution
        print(f"    WARNING: Status={LpStatus[prob.status]}")

    # Extract loading solution
    L_sol = {}
    for node in NODES:
        for s_idx in range(len(STEPS[node])):
            for fab in FABS:
                mt_val = max(0, value(L_mt[(node, s_idx, fab)]) or 0)
                tor_val = (
                    max(0, value(L_tor.get((node, s_idx, fab), 0)) or 0)
                    if not is_q1
                    else 0
                )
                L_sol[(node, s_idx, fab)] = round(mt_val + tor_val)

    # Extract tool counts
    if is_q1:
        T_sol = {ws: list(prev_tools[ws]) for ws in WS_ALL}
    else:
        T_sol = {
            ws: [max(0, round(value(T[ws][f]) or 0)) for f in range(3)] for ws in WS_ALL
        }

    cost = value(prob.objective) or 0
    return L_sol, T_sol, cost


# ============================================================
# MAIN
# ============================================================
def solve_all():
    print("=" * 80)
    print("Q1b CORRECT LP SOLVER — Step-Level Loading")
    print("=" * 80)

    t0 = time_module.time()
    tools_plan = []
    flow_plan = []
    total_cost = 0.0
    prev_tools = Q1_TOOLS

    for q_idx in range(8):
        L_sol, T_sol, cost = solve_quarter(q_idx, prev_tools, verbose=True)
        tools_plan.append(T_sol)
        flow_plan.append(L_sol)
        total_cost += cost
        prev_tools = T_sol

        # Summary
        for node in NODES:
            s0 = [L_sol.get((node, 0, fab), 0) for fab in FABS]
            print(
                f"    N{node} S1: F1={s0[0]}, F2={s0[1]}, F3={s0[2]}, total={sum(s0)}"
            )
        sp = [sum(T_sol[ws][f] * SPACE[ws] for ws in WS_ALL) for f in range(3)]
        print(
            f"    Space: F1={sp[0]:.0f}/{FAB_SPACE[1]}, F2={sp[1]:.0f}/{FAB_SPACE[2]}, F3={sp[2]:.0f}/{FAB_SPACE[3]}"
        )

    print(f"\nGRAND TOTAL (K3): ${total_cost / 1e9:.4f}B  (${total_cost / 1e6:.1f}M)")
    print(f"Runtime: {time_module.time() - t0:.1f}s")

    # ---- VALIDATION ----
    print("\nVALIDATION:")
    all_ok = True
    for q_idx in range(8):
        quarter = QUARTERS[q_idx]
        tools = tools_plan[q_idx]
        flow = flow_plan[q_idx]
        errs = []

        # Demand check
        for node in NODES:
            for s_idx in range(len(STEPS[node])):
                total = sum(flow.get((node, s_idx, fab), 0) for fab in FABS)
                if abs(total - DEMAND[node][q_idx]) > 1:
                    errs.append(f"N{node}S{s_idx + 1}:{total}!={DEMAND[node][q_idx]}")

        # Tool capacity check
        for ws in WS_ALL:
            is_tor = ws.endswith("+")
            for fab_idx, fab in enumerate(FABS):
                avail = tools[ws][fab_idx]
                req = 0
                for node in NODES:
                    for s_idx, (ws_mt, rpt_mt, ws_t, rpt_t) in enumerate(STEPS[node]):
                        loading = flow.get((node, s_idx, fab), 0)
                        if not is_tor and ws_mt == ws:
                            req += loading * rpt_mt / (MIN_PER_WEEK * UTIL[ws])
                        elif is_tor and ws_t == ws:
                            req += loading * rpt_t / (MIN_PER_WEEK * UTIL[ws])
                if req > avail + 0.5:
                    errs.append(f"{ws}F{fab}:{req:.1f}>{avail}")

        # Space check
        for fab_idx, fab in enumerate(FABS):
            sp = sum(tools[ws][fab_idx] * SPACE[ws] for ws in WS_ALL)
            if sp > FAB_SPACE[fab] + 0.5:
                errs.append(f"SpF{fab}:{sp:.0f}>{FAB_SPACE[fab]}")

        status = "OK" if not errs else f"FAIL: {errs[:3]}"
        print(f"  {quarter}: {status}")
        if errs:
            all_ok = False

    if all_ok:
        print("\nALL CONSTRAINTS SATISFIED!")

    # Save
    result = {
        "grand_total": total_cost,
        "quarters": QUARTERS,
        "tools_plan": [
            {"quarter": QUARTERS[q], "tools": {ws: tools_plan[q][ws] for ws in WS_ALL}}
            for q in range(8)
        ],
        "flow_plan": [
            {
                "quarter": QUARTERS[q],
                "flow": [
                    {
                        "node": n,
                        "step": s + 1,
                        "fab": f,
                        "loading": flow_plan[q].get((n, s, f), 0),
                    }
                    for n in NODES
                    for s in range(len(STEPS[n]))
                    for f in FABS
                ],
            }
            for q in range(8)
        ],
    }
    out_path = _ROOT / "results" / "q1b_correct_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved: {out_path}")

    return tools_plan, flow_plan, total_cost


if __name__ == "__main__":
    solve_all()
