"""
Script 8: Fill Excel Answer Sheet (Part A & Part B)
===================================================
Fills the Q1a or Q1b answer sheet using the corresponding results JSON.
Prompts the user to choose which part to fill (A or B). Part A pulls from
results/q1a_results.json; Part B pulls from results/q1b_solution_v2_results.json.

Answer areas:
1. Tool count section: cols C, D, E (Fab1, Fab2, Fab3 WS counts) for rows 8-103
    - Each quarter has 12 rows: 6 mintech (A-F) + 6 TOR (A+-F+)
    - Q1'26: rows 8-19, Q2'26: rows 20-31, Q3'26: rows 32-43, Q4'26: rows 44-55
    - Q1'27: rows 56-67, Q2'27: rows 68-79, Q3'27: rows 80-91, Q4'27: rows 92-103

2. Flow section: col S for rows 8-1039
    - Each row: (quarter, node, step, fab) -> loading value
    - Part A: aggregate per-(quarter,node,fab) from first step (mintech+TOR)
    - Part B: use assignment per (quarter,node,fab) directly from the Part B JSON
"""

import json
import shutil
from pathlib import Path

import openpyxl

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & LOAD SOLUTIONS
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SRC_SHEET = ROOT / "5)BACC2026ParticipantAnswerSheet.xlsx"
DST_SHEET = RESULTS_DIR / "BACC2026_FilledAnswerSheet.xlsx"
Q1A_RESULTS_PATH = RESULTS_DIR / "q1a_results.json"
Q1B_RESULTS_PATH = RESULTS_DIR / "q1b_solution_v2_results.json"

with open(Q1A_RESULTS_PATH) as f:
    SOL_A = json.load(f)

with open(Q1B_RESULTS_PATH) as f:
    SOL_B = json.load(f)

QUARTERS = ["Q1'26", "Q2'26", "Q3'26", "Q4'26", "Q1'27", "Q2'27", "Q3'27", "Q4'27"]
FABS = [1, 2, 3]
NODES = [1, 2, 3]
MINTECH_WS = ["A", "B", "C", "D", "E", "F"]
TOR_WS = ["A+", "B+", "C+", "D+", "E+", "F+"]
ALL_WS = MINTECH_WS + TOR_WS


def safe_get(dic, *keys, default=0):
    cur = dic
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k)
    return default if cur is None else cur


# Tool counts: {quarter: {ws: {fab: count}}}
def get_tool_count_a(q, ws, f):
    q = normalize_q(q)
    return safe_get(SOL_A, "total_tools", q, ws, str(f), default=0)


def get_tool_count_b(q, ws, f):
    q = normalize_q(q)
    if ws in TOR_WS:
        return safe_get(SOL_B, "tor_tools", q, ws, str(f), default=0)
    return safe_get(SOL_B, "mt_tools", q, ws, str(f), default=0)


# Quarter key normalization: Excel uses Unicode right single quote \u2019, JSON uses ASCII apostrophe
def normalize_q(q):
    return q.replace("\u2019", "'").replace("\u2018", "'")


# Flow (aggregate per quarter/node/fab) for Part A: use first step's mintech+TOR flow
def get_flow_a(q, n, f):
    q_norm = normalize_q(q)
    flow_mt_q = safe_get(SOL_A, "flow_mt", q_norm, str(n), default={})
    flow_tor_q = safe_get(SOL_A, "flow_tor", q_norm, str(n), default={})

    def first_step(flow_dict):
        if not flow_dict:
            return None
        try:
            return min(flow_dict.keys(), key=lambda s: int(s))
        except Exception:
            return next(iter(flow_dict.keys()))

    step_key = first_step(flow_mt_q) or first_step(flow_tor_q)
    if step_key is None:
        return 0

    mt_val = safe_get(flow_mt_q, step_key, str(f), default=0)
    tor_val = safe_get(flow_tor_q, step_key, str(f), default=0)
    return mt_val + tor_val


# Flow for Part B: assignment per (quarter,node,fab)
def get_flow_b(q, n, f):
    q_norm = normalize_q(q)
    return safe_get(SOL_B, "assignment", q_norm, str(n), str(f), default=0)


# ─────────────────────────────────────────────────────────────────────────────
# FILL SHEET (generic)
# ─────────────────────────────────────────────────────────────────────────────


def fill_sheet(wb, sheet_name, part):
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
                if part == "A":
                    count = get_tool_count_a(q, tool_ws, fab)
                else:
                    count = get_tool_count_b(q, tool_ws, fab)
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
            if part == "A":
                flow = get_flow_a(q_val, int(n_val), int(f_val))
            else:
                flow = get_flow_b(q_val, int(n_val), int(f_val))
            ws.cell(r, 19).value = flow
            flow_filled += 1

    print(f"  Filled {flow_filled} flow cells")

    return tools_filled + flow_filled


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    choice = input("Which part to fill? Enter A or B: ").strip().upper()
    if choice not in {"A", "B"}:
        raise SystemExit("Invalid choice. Please enter 'A' or 'B'.")

    sheet_name = "Q1a" if choice == "A" else "Q1b"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_SHEET, DST_SHEET)
    print(f"Copied answer sheet to: {DST_SHEET}")

    wb = openpyxl.load_workbook(DST_SHEET)
    print(f"Sheets: {wb.sheetnames}")

    if sheet_name in wb.sheetnames:
        print(f"\nFilling {sheet_name} sheet (Part {choice} results)...")
        n = fill_sheet(wb, sheet_name, choice)
        print(f"  Total cells filled: {n}")
    else:
        raise SystemExit(f"Sheet {sheet_name} not found in workbook.")

    wb.save(DST_SHEET)
    print(f"\nSaved filled answer sheet to: {DST_SHEET}")

    # Verify by reading back (first block only)
    print("\n=== Verification ===")
    wb2 = openpyxl.load_workbook(DST_SHEET)
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
