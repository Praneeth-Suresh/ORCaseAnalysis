"""
Script 1: Data Extraction and Problem Parameter Validation
==========================================================
Reads all relevant data from the BACC 2026 Excel answer sheet and
structures it for use in the optimization models.
"""

import json
import math
from pathlib import Path

import openpyxl

ROOT_DIR = Path(__file__).resolve().parent
PARAMETERS_DIR = ROOT_DIR / "parameters"
PARAMS_PATH = PARAMETERS_DIR / "params.json"

# ─────────────────────────────────────────────────────────────────────────────
# HARD-CODED PROBLEM DATA (from the question paper)
# ─────────────────────────────────────────────────────────────────────────────

QUARTERS = ["Q1'26", "Q2'26", "Q3'26", "Q4'26", "Q1'27", "Q2'27", "Q3'27", "Q4'27"]
FABS = [1, 2, 3]
NODES = [1, 2, 3]

# Weekly wafer loading per node per quarter
LOADING = {
    1: {
        "Q1'26": 12000,
        "Q2'26": 10000,
        "Q3'26": 8500,
        "Q4'26": 7500,
        "Q1'27": 6000,
        "Q2'27": 5000,
        "Q3'27": 4000,
        "Q4'27": 2000,
    },
    2: {
        "Q1'26": 5000,
        "Q2'26": 5200,
        "Q3'26": 5400,
        "Q4'26": 5600,
        "Q1'27": 6000,
        "Q2'27": 6500,
        "Q3'27": 7000,
        "Q4'27": 7500,
    },
    3: {
        "Q1'26": 3000,
        "Q2'26": 4500,
        "Q3'26": 7000,
        "Q4'26": 8000,
        "Q1'27": 9000,
        "Q2'27": 11000,
        "Q3'27": 13000,
        "Q4'27": 16000,
    },
}

# Process steps: node -> list of (step_num, ws_mintech, rpt_mintech, ws_tor, rpt_tor)
PROCESS_STEPS = {
    1: [
        (1, "D", 14, "D+", 12),
        (2, "F", 25, "F+", 21),
        (3, "F", 27, "F+", 23),
        (4, "A", 20, "A+", 16),
        (5, "F", 12, "F+", 9),
        (6, "D", 27, "D+", 21),
        (7, "D", 17, "D+", 13),
        (8, "A", 18, "A+", 16),
        (9, "A", 16, "A+", 13),
        (10, "D", 14, "D+", 11),
        (11, "F", 18, "F+", 16),
    ],
    2: [
        (1, "F", 19, "F+", 16),
        (2, "B", 20, "B+", 18),
        (3, "E", 10, "E+", 7),
        (4, "B", 25, "B+", 19),
        (5, "B", 15, "B+", 11),
        (6, "F", 16, "F+", 14),
        (7, "F", 17, "F+", 15),
        (8, "B", 22, "B+", 16),
        (9, "E", 7, "E+", 6),
        (10, "E", 9, "E+", 7),
        (11, "E", 20, "E+", 19),
        (12, "F", 21, "F+", 18),
        (13, "E", 12, "E+", 9),
        (14, "E", 15, "E+", 12),
        (15, "E", 13, "E+", 10),
    ],
    3: [
        (1, "C", 21, "C+", 20),
        (2, "D", 9, "D+", 7),
        (3, "E", 24, "E+", 23),
        (4, "E", 15, "E+", 11),
        (5, "F", 16, "F+", 14),
        (6, "D", 12, "D+", 11),
        (7, "C", 24, "C+", 21),
        (8, "C", 19, "C+", 13),
        (9, "D", 15, "D+", 13),
        (10, "D", 24, "D+", 20),
        (11, "E", 17, "E+", 15),
        (12, "E", 18, "E+", 13),
        (13, "F", 20, "F+", 18),
        (14, "C", 12, "C+", 11),
        (15, "D", 11, "D+", 10),
        (16, "C", 25, "C+", 20),
        (17, "F", 14, "F+", 13),
    ],
}

# Workstation specifications: ws -> (space_m2, capex_per_tool, utilization)
WS_SPECS = {
    "A": (6.78, 4_500_000, 0.78),
    "B": (3.96, 6_000_000, 0.76),
    "C": (5.82, 2_200_000, 0.80),
    "D": (5.61, 4_000_000, 0.80),
    "E": (4.65, 3_500_000, 0.76),
    "F": (3.68, 6_000_000, 0.80),
    "A+": (6.93, 6_000_000, 0.84),
    "B+": (3.72, 8_000_000, 0.81),
    "C+": (5.75, 3_200_000, 0.86),
    "D+": (5.74, 5_500_000, 0.88),
    "E+": (4.80, 5_800_000, 0.84),
    "F+": (3.57, 8_000_000, 0.90),
}

# Fab specifications: fab -> (total_space_m2, {ws: initial_count})
FAB_SPECS = {
    1: {
        "space": 1500,
        "tools": {
            "A": 50,
            "B": 25,
            "C": 0,
            "D": 50,
            "E": 40,
            "F": 90,
            "A+": 0,
            "B+": 0,
            "C+": 0,
            "D+": 0,
            "E+": 0,
            "F+": 0,
        },
    },
    2: {
        "space": 1300,
        "tools": {
            "A": 35,
            "B": 30,
            "C": 0,
            "D": 50,
            "E": 30,
            "F": 60,
            "A+": 0,
            "B+": 0,
            "C+": 0,
            "D+": 0,
            "E+": 0,
            "F+": 0,
        },
    },
    3: {
        "space": 700,
        "tools": {
            "A": 0,
            "B": 0,
            "C": 40,
            "D": 35,
            "E": 16,
            "F": 36,
            "A+": 0,
            "B+": 0,
            "C+": 0,
            "D+": 0,
            "E+": 0,
            "F+": 0,
        },
    },
}

# Cost parameters
TRANSFER_COST_PER_WAFER = 50  # $ per wafer per transfer
MOVEOUT_COST_PER_TOOL = 1_000_000  # $1M per tool

# Time constants
WEEKS_PER_QUARTER = 13
MINUTES_PER_WEEK = 7 * 24 * 60  # 10,080 minutes


# ─────────────────────────────────────────────────────────────────────────────
# DERIVED PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────


def compute_tool_requirement(
    loading_per_week: float, rpt_min: float, utilization: float
) -> float:
    """
    Tool Requirement = Loading * RPT / (7 * 24 * 60 * Utilization)
    Returns the fractional tool requirement; must be rounded up (ceiling) to get integer count.
    """
    return (loading_per_week * rpt_min) / (MINUTES_PER_WEEK * utilization)


def get_ws_for_step(node: int, step: int) -> dict:
    """Returns {'mintech': ws, 'rpt_mintech': rpt, 'tor': ws_tor, 'rpt_tor': rpt_tor}."""
    for s, ws, rpt, ws_tor, rpt_tor in PROCESS_STEPS[node]:
        if s == step:
            return {
                "mintech": ws,
                "rpt_mintech": rpt,
                "tor": ws_tor,
                "rpt_tor": rpt_tor,
            }
    raise ValueError(f"Step {step} not found for Node {node}")


def get_initial_space_used(fab: int) -> float:
    """Returns the total floor space used by initial tools in a fab."""
    total = 0.0
    for ws, count in FAB_SPECS[fab]["tools"].items():
        space, _, _ = WS_SPECS[ws]
        total += space * count
    return total


def validate_initial_space():
    """Checks that the initial tool configuration does not exceed fab space limits."""
    print("\n=== Initial Space Validation ===")
    all_ok = True
    for fab in FABS:
        used = get_initial_space_used(fab)
        available = FAB_SPECS[fab]["space"]
        status = "OK" if used <= available else "EXCEEDED"
        if used > available:
            all_ok = False
        print(f"  Fab {fab}: Used={used:.2f} m², Available={available} m²  [{status}]")
    return all_ok


def compute_aggregate_rpt_by_ws():
    """
    For each node, compute the total RPT per wafer for each workstation type.
    This helps identify which workstations are the bottlenecks.
    """
    print("\n=== Aggregate RPT per Wafer by Workstation (minutes) ===")
    ws_list = ["A", "B", "C", "D", "E", "F"]
    header = f"{'Node':<8}" + "".join(f"{ws:<8}" for ws in ws_list)
    print(header)
    for node in NODES:
        rpt_totals = {ws: 0 for ws in ws_list}
        for step_num, ws, rpt, ws_tor, rpt_tor in PROCESS_STEPS[node]:
            rpt_totals[ws] += rpt
        row = f"Node {node}  " + "".join(f"{rpt_totals[ws]:<8}" for ws in ws_list)
        print(row)


def compute_peak_tool_requirements():
    """
    For each workstation, compute the peak tool requirement across all quarters
    assuming all production runs on mintech tools in a single fab.
    This gives a lower bound on the total tools needed.
    """
    print("\n=== Peak Tool Requirements (single-fab, all-mintech, per workstation) ===")
    ws_list = ["A", "B", "C", "D", "E", "F"]
    print(f"{'Quarter':<10}" + "".join(f"{ws:<10}" for ws in ws_list))
    for q in QUARTERS:
        req = {ws: 0.0 for ws in ws_list}
        for node in NODES:
            loading = LOADING[node][q]
            for step_num, ws, rpt, ws_tor, rpt_tor in PROCESS_STEPS[node]:
                _, _, util = WS_SPECS[ws]
                req[ws] += compute_tool_requirement(loading, rpt, util)
        row = f"{q:<10}" + "".join(f"{math.ceil(req[ws]):<10}" for ws in ws_list)
        print(row)


def compute_available_capacity_by_fab():
    """
    For each fab, compute the available tool capacity per workstation.
    """
    print("\n=== Available Tools per Fab (initial Q1'26) ===")
    ws_list = ["A", "B", "C", "D", "E", "F"]
    print(f"{'Fab':<8}" + "".join(f"{ws:<8}" for ws in ws_list) + "  Space(m²)")
    for fab in FABS:
        tools = FAB_SPECS[fab]["tools"]
        row = f"Fab {fab}   " + "".join(f"{tools[ws]:<8}" for ws in ws_list)
        row += f"  {get_initial_space_used(fab):.1f} / {FAB_SPECS[fab]['space']}"
        print(row)


def check_fab_capability():
    """
    For each fab, determine which workstations it has (initial tools > 0).
    This determines which steps can be run in which fab without new purchases.
    """
    print("\n=== Fab Capability (workstations with initial tools > 0) ===")
    ws_list = ["A", "B", "C", "D", "E", "F"]
    for fab in FABS:
        capable_ws = [ws for ws in ws_list if FAB_SPECS[fab]["tools"][ws] > 0]
        print(f"  Fab {fab}: {capable_ws}")


def export_parameters_json(output_path: str):
    """Exports all parameters to a JSON file for use by solver scripts."""
    params = {
        "quarters": QUARTERS,
        "fabs": FABS,
        "nodes": NODES,
        "loading": LOADING,
        "process_steps": {
            str(n): [
                {
                    "step": s,
                    "ws_mintech": ws,
                    "rpt_mintech": rpt,
                    "ws_tor": ws_tor,
                    "rpt_tor": rpt_tor,
                }
                for s, ws, rpt, ws_tor, rpt_tor in steps
            ]
            for n, steps in PROCESS_STEPS.items()
        },
        "ws_specs": {
            ws: {"space_m2": space, "capex": capex, "utilization": util}
            for ws, (space, capex, util) in WS_SPECS.items()
        },
        "fab_specs": {
            str(f): {"space": spec["space"], "tools": spec["tools"]}
            for f, spec in FAB_SPECS.items()
        },
        "transfer_cost_per_wafer": TRANSFER_COST_PER_WAFER,
        "moveout_cost_per_tool": MOVEOUT_COST_PER_TOOL,
        "weeks_per_quarter": WEEKS_PER_QUARTER,
        "minutes_per_week": MINUTES_PER_WEEK,
    }
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nParameters exported to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PARAMETERS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BACC 2026 – Data Extraction & Validation Report")
    print("=" * 60)

    # 1. Validate initial space
    ok = validate_initial_space()
    if not ok:
        print("  WARNING: Initial tool configuration exceeds space limits!")

    # 2. Show fab capability
    check_fab_capability()

    # 3. Show aggregate RPT by workstation
    compute_aggregate_rpt_by_ws()

    # 4. Show available capacity
    compute_available_capacity_by_fab()

    # 5. Show peak tool requirements
    compute_peak_tool_requirements()

    # 6. Export parameters
    export_parameters_json(PARAMS_PATH)

    print("\n" + "=" * 60)
    print("Data extraction complete. Ready for optimization.")
    print("=" * 60)
