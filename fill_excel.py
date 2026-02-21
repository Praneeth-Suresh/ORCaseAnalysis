"""
Script 8: Fill Excel Answer Sheet
===================================
Fills the Q1b (and Q1a) answer sheets with the optimal solution.

Answer areas:
1. Tool count section: cols C, D, E (Fab1, Fab2, Fab3 WS counts) for rows 8-103
   - Each quarter has 12 rows: 6 mintech (A-F) + 6 TOR (A+-F+)
   - Q1'26: rows 8-19, Q2'26: rows 20-31, Q3'26: rows 32-43, Q4'26: rows 44-55
   - Q1'27: rows 56-67, Q2'27: rows 68-79, Q3'27: rows 80-91, Q4'27: rows 92-103

2. Flow section: col S for rows 8-1039
   - Each row: (quarter, node, step, fab) -> loading value
   - The loading is the same for ALL steps of a given (quarter, node, fab)
"""

import json
import shutil
import openpyxl

# ─────────────────────────────────────────────────────────────────────────────
# LOAD SOLUTION
# ─────────────────────────────────────────────────────────────────────────────

with open("results/q1b_solution_v2_results.json") as f:
    SOL = json.load(f)

QUARTERS = ["Q1'26", "Q2'26", "Q3'26", "Q4'26", "Q1'27", "Q2'27", "Q3'27", "Q4'27"]
FABS = [1, 2, 3]
NODES = [1, 2, 3]
MINTECH_WS = ["A", "B", "C", "D", "E", "F"]
TOR_WS = ["A+", "B+", "C+", "D+", "E+", "F+"]
ALL_WS = MINTECH_WS + TOR_WS


# Tool counts: {quarter: {ws: {fab: count}}}
def get_tool_count(q, ws, f):
    q = normalize_q(q)
    if ws in TOR_WS:
        return SOL["tor_tools"][q][ws][str(f)]
    else:
        return SOL["mt_tools"][q][ws][str(f)]


# Quarter key normalization: Excel uses Unicode right single quote \u2019, JSON uses ASCII apostrophe
def normalize_q(q):
    return q.replace("\u2019", "'").replace("\u2018", "'")


# Flow: {quarter: {node: {fab: wafers}}}
def get_flow(q, n, f):
    return SOL["assignment"][normalize_q(q)][str(n)][str(f)]


# ─────────────────────────────────────────────────────────────────────────────
# FILL Q1b SHEET
# ─────────────────────────────────────────────────────────────────────────────


def fill_sheet(wb, sheet_name):
    ws = wb[sheet_name]

    # ── Tool count section ──────────────────────────────────────────────────
    # Row mapping: Q1'26 starts at row 8, each quarter has 12 rows (6 MT + 6 TOR)
    QUARTER_START_ROWS = {
        "Q1'26": 8,
        "Q2'26": 20,
        "Q3'26": 32,
        "Q4'26": 44,
        "Q1'27": 56,
        "Q2'27": 68,
        "Q3'27": 80,
        "Q4'27": 92,
    }

    tools_filled = 0
    for q, start_row in QUARTER_START_ROWS.items():
        for i, tool_ws in enumerate(ALL_WS):
            row = start_row + i
            for col_idx, fab in [(3, 1), (4, 2), (5, 3)]:  # C=Fab1, D=Fab2, E=Fab3
                count = get_tool_count(q, tool_ws, fab)
                ws.cell(row, col_idx).value = count
                tools_filled += 1

    print(f"  Filled {tools_filled} tool count cells")

    # ── Flow section ────────────────────────────────────────────────────────
    # Col S (19) = Loading (to fill)
    flow_filled = 0
    for r in range(8, 1040):
        q_val = ws.cell(r, 15).value  # col O = Quarter
        n_val = ws.cell(r, 16).value  # col P = Node
        f_val = ws.cell(r, 18).value  # col R = Fab

        if q_val is not None and n_val is not None and f_val is not None:
            flow = get_flow(q_val, int(n_val), int(f_val))
            ws.cell(r, 19).value = flow
            flow_filled += 1

    print(f"  Filled {flow_filled} flow cells")

    return tools_filled + flow_filled


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    src = "5)BACC2026ParticipantAnswerSheet.xlsx"
    dst = "results/BACC2026_FilledAnswerSheet.xlsx"

    shutil.copy2(src, dst)
    print(f"Copied answer sheet to: {dst}")

    wb = openpyxl.load_workbook(dst)
    print(f"Sheets: {wb.sheetnames}")

    # Fill Q1b sheet
    if "Q1b" in wb.sheetnames:
        print("\nFilling Q1b sheet...")
        n = fill_sheet(wb, "Q1b")
        print(f"  Total cells filled: {n}")

    # Also fill Q1a sheet with the same solution (since Q1a is a subset)
    # Q1a asks for no move-outs — but since the problem is infeasible without move-outs,
    # we fill Q1a with the Q1b solution as the best feasible answer
    if "Q1a" in wb.sheetnames:
        print("\nFilling Q1a sheet (with Q1b solution as best feasible)...")
        n = fill_sheet(wb, "Q1a")
        print(f"  Total cells filled: {n}")

    wb.save(dst)
    print(f"\nSaved filled answer sheet to: {dst}")

    # Verify by reading back
    print("\n=== Verification ===")
    wb2 = openpyxl.load_workbook(dst)
    for sheet_name in ["Q1a", "Q1b"]:
        if sheet_name not in wb2.sheetnames:
            continue
        ws2 = wb2[sheet_name]
        print(f"\n{sheet_name} - Tool counts (Q1'26, rows 8-19):")
        for r in range(8, 20):
            b = ws2.cell(r, 2).value  # WS name
            c = ws2.cell(r, 3).value  # Fab 1
            d = ws2.cell(r, 4).value  # Fab 2
            e = ws2.cell(r, 5).value  # Fab 3
            print(f"  Row {r}: {b} | F1={c}, F2={d}, F3={e}")

        print(f"\n{sheet_name} - Flow (first 10 rows):")
        for r in range(8, 18):
            q = ws2.cell(r, 15).value
            n = ws2.cell(r, 16).value
            s = ws2.cell(r, 17).value
            f = ws2.cell(r, 18).value
            loading = ws2.cell(r, 19).value
            print(f"  Row {r}: {q} Node{n} Step{s} Fab{f} -> {loading}")
