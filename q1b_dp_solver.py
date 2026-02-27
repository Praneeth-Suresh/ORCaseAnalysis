"""
Script: Q1b Dynamic Programming Solver (Parallelized)
======================================================
Attempts to find the globally optimal solution to Q1b using a
multi-stage Dynamic Programming approach with parallelized
state-space exploration.

PROBLEM STRUCTURE FOR DP
--------------------------
The problem has a natural stage structure:
  - Stages: 8 quarters (Q1'26 ... Q4'27)
  - State at stage q: (mt_tools, tor_tools) — the tool inventory
    across all fabs at the END of quarter q.
  - Decision at stage q: (assignment, buy_tor, moveout) — how many
    wafers of each node go to each fab, how many TOR tools to buy,
    and how many mintech tools to move out.
  - Transition: state_{q+1} = state_q + buy_tor - moveout
  - Cost: CapEx(buy_tor) + OpEx(moveout)

STATE SPACE ANALYSIS
---------------------
The full state is (mt_counts[6 ws × 3 fabs], tor_counts[6 ws × 3 fabs]).
This is 36 integers — far too large for exact DP.

TRACTABLE DECOMPOSITION
------------------------
Key insight: because all production uses TOR tools (proven optimal),
the mintech tools are "dead weight" that only consume space. The
optimal strategy is to move them out as fast as needed. The only
real decision is the NODE-FAB ASSIGNMENT.

The DP state is therefore:
  state = (w1_f1, w1_f2, w3_f1, w3_f2) — the flow of Nodes 1 and 3
  in Fabs 1 and 2 (Node 2 fills Fab 2, Node 3 fills Fab 3 first).

This gives a 4-dimensional state space. With discretisation to
100-wafer granularity, each dimension has ~200 values → 200^4 = 1.6B
states — still too large.

FURTHER REDUCTION: INTER-QUARTER INDEPENDENCE
-----------------------------------------------
The key observation is that the cost function DECOMPOSES across
quarters: the cost in quarter q depends only on the INCREMENTAL
tool purchases (buy_tor[q] = max(0, tor_req[q] - tor_req[q-1])).

This means the DP reduces to a SHORTEST PATH problem on a DAG:
  - Each node is a feasible assignment for a quarter
  - Each edge cost is the incremental CapEx + moveout OpEx
  - We want the minimum-cost path from Q1'26 to Q4'27

The state is the TOR tool inventory at the end of each quarter,
which determines the incremental purchase cost for the next quarter.

PARALLELIZATION STRATEGY
--------------------------
For each quarter, we enumerate all feasible assignments in parallel
using multiprocessing.Pool. Each worker evaluates one candidate
assignment and returns its cost and tool requirements.

Author: Manus AI
Date: 2026-02-22
"""

import json
import math
import time
import itertools
from multiprocessing import Pool, cpu_count
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

with open("parameters/params.json") as f:
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
INITIAL_MT = {ws: {f: FAB_SPECS[f]["tools"][ws] for f in FABS} for ws in MINTECH_WS}

FAB_SPACE = {f: FAB_SPECS[f]["space"] for f in FABS}  # {1:1500, 2:1300, 3:700}

# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS (module-level for multiprocessing pickling)
# ─────────────────────────────────────────────────────────────────────────────


def compute_tor_tools(flow_nf):
    """
    Given flow_nf = {n: wafers} for a single fab,
    returns (space_used, tor_counts_dict).
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


def tor_tools_tuple(flow_nf):
    """Returns TOR tool counts as a tuple (for hashing)."""
    _, tor_int = compute_tor_tools(flow_nf)
    return tuple(tor_int[ws] for ws in TOR_WS)


def compute_incremental_capex(prev_tor_tuple, new_tor_tuple):
    """Cost of buying new TOR tools = sum of (new - prev) * capex for each WS."""
    cost = 0
    for i, ws in enumerate(TOR_WS):
        delta = max(0, new_tor_tuple[i] - prev_tor_tuple[i])
        cost += delta * WS_SPECS[ws][1]
    return cost


def compute_moveout_cost(tor_space_per_fab, mt_counts):
    """
    Given TOR space needed per fab and current MT tool counts,
    compute the minimum move-out cost to free enough space.
    Returns (moveout_cost, new_mt_counts).
    """
    total_cost = 0
    new_mt = {ws: {f: mt_counts[ws][f] for f in FABS} for ws in MINTECH_WS}

    for f in FABS:
        space_mt = sum(WS_SPECS[ws][0] * new_mt[ws][f] for ws in MINTECH_WS)
        excess = tor_space_per_fab[f] + space_mt - FAB_SPACE[f]
        if excess > 0.001:
            # Move out largest-footprint tools first
            ws_sorted = sorted(
                [(ws, new_mt[ws][f]) for ws in MINTECH_WS if new_mt[ws][f] > 0],
                key=lambda x: -WS_SPECS[x[0]][0],
            )
            for ws, count in ws_sorted:
                if excess <= 0.001:
                    break
                sp = WS_SPECS[ws][0]
                to_move = min(count, math.ceil(excess / sp))
                new_mt[ws][f] -= to_move
                total_cost += to_move * MOVEOUT_COST
                excess -= to_move * sp

    return total_cost, new_mt


def evaluate_assignment(args):
    """
    Worker function: evaluate a single (quarter, assignment) candidate.
    Returns (assignment_tuple, total_tor_tuple, tor_space_per_fab, feasible).
    """
    q_idx, assignment_dict = args
    q = QUARTERS[q_idx]

    # assignment_dict: {n: {f: wafers}}
    # Check demand
    for n in NODES:
        total = sum(assignment_dict[n][f] for f in FABS)
        if total != LOADING[n][q]:
            return None  # infeasible

    # Compute TOR tools per fab
    tor_per_fab = {}
    tor_space_per_fab = {}
    for f in FABS:
        flow_nf = {n: assignment_dict[n][f] for n in NODES}
        space, tor_int = compute_tor_tools(flow_nf)
        tor_per_fab[f] = tor_int
        tor_space_per_fab[f] = space
        if space > FAB_SPACE[f]:
            return None  # infeasible (space exceeded even with no MT tools)

    # Encode as tuple: (f1_tor_tuple, f2_tor_tuple, f3_tor_tuple)
    tor_tuple = tuple(tuple(tor_per_fab[f][ws] for ws in TOR_WS) for f in FABS)

    return (assignment_dict, tor_tuple, tor_space_per_fab)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE CANDIDATE ASSIGNMENTS FOR A QUARTER
# ─────────────────────────────────────────────────────────────────────────────


def generate_assignments(q, granularity=500):
    """
    Generate all feasible assignments for quarter q at the given granularity.

    The assignment is parameterized by:
      - w1_f1: Node 1 wafers in Fab 1 (rest go to Fab 2)
      - w2_f2: Node 2 wafers in Fab 2 (rest go to Fab 1)
      - w3_f3: Node 3 wafers in Fab 3 (fixed at max feasible)
      - w3_f1: Node 3 wafers in Fab 1 (rest go to Fab 2)

    We do not assign Node 1 to Fab 3 or Node 2 to Fab 3 (Fab 3 only has C tools,
    which are only used by Node 3).
    """
    D1 = LOADING[1][q]
    D2 = LOADING[2][q]
    D3 = LOADING[3][q]

    # Max Node 3 in Fab 3 (space-limited)
    max_n3_f3 = 0
    for w in range(0, D3 + granularity, granularity):
        sp, _ = compute_tor_tools({3: w})
        if sp <= FAB_SPACE[3]:
            max_n3_f3 = w
        else:
            break
    n3_f3 = min(D3, max_n3_f3)
    n3_remaining = D3 - n3_f3

    assignments = []

    # Enumerate w1_f1 (Node 1 in Fab 1)
    for w1_f1 in range(0, D1 + granularity, granularity):
        if w1_f1 > D1:
            w1_f1 = D1
        w1_f2 = D1 - w1_f1

        # Enumerate w2_f2 (Node 2 in Fab 2)
        for w2_f2 in range(0, D2 + granularity, granularity):
            if w2_f2 > D2:
                w2_f2 = D2
            w2_f1 = D2 - w2_f2

            # Enumerate w3_f1 (Node 3 in Fab 1, from remaining after Fab 3)
            for w3_f1 in range(0, n3_remaining + granularity, granularity):
                if w3_f1 > n3_remaining:
                    w3_f1 = n3_remaining
                w3_f2 = n3_remaining - w3_f1

                assignment = {
                    1: {1: w1_f1, 2: w1_f2, 3: 0},
                    2: {1: w2_f1, 2: w2_f2, 3: 0},
                    3: {1: w3_f1, 2: w3_f2, 3: n3_f3},
                }
                assignments.append((QUARTERS.index(q), assignment))

                if w3_f1 == n3_remaining:
                    break
            if w2_f2 == D2:
                break
        if w1_f1 == D1:
            break

    return assignments


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC PROGRAMMING SOLVER
# ─────────────────────────────────────────────────────────────────────────────


def dp_solve(granularity=500, n_workers=None):
    """
    Multi-stage DP with parallelized state evaluation.

    State: (mt_counts_tuple) — the remaining mintech tool counts
           (TOR tools are determined by the assignment, not carried forward
            since we use just-in-time purchasing)

    At each stage, we:
    1. Generate all feasible assignments (parallelized)
    2. For each assignment, compute the TOR tools needed
    3. Compute the incremental CapEx (vs previous stage TOR tools)
    4. Compute the move-out cost (to free space for TOR tools)
    5. Store the minimum-cost path to each state

    Returns: (best_cost, best_path)
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"\nDP Solver: granularity={granularity} wafers, workers={n_workers}")
    print(f"State space: MT tool counts across 6 WS × 3 fabs")
    print()

    # Initial state: initial MT tool counts
    def mt_to_tuple(mt_counts):
        return tuple(mt_counts[ws][f] for ws in MINTECH_WS for f in FABS)

    def tuple_to_mt(t):
        mt = {}
        idx = 0
        for ws in MINTECH_WS:
            mt[ws] = {}
            for f in FABS:
                mt[ws][f] = t[idx]
                idx += 1
        return mt

    # Initial TOR state: all zeros (no TOR tools initially)
    def tor_to_tuple(tor_per_fab):
        return tuple(tor_per_fab[f][ws] for f in FABS for ws in TOR_WS)

    zero_tor = tuple(0 for _ in range(len(FABS) * len(TOR_WS)))

    initial_mt_tuple = mt_to_tuple(INITIAL_MT)

    # DP table: {state: (cost, path)}
    # state = (mt_tuple, tor_tuple)
    # We keep only the Pareto-optimal states (min cost per state)

    # Stage 0: initial state
    dp = {(initial_mt_tuple, zero_tor): (0, [])}

    start_time = time.time()

    for q_idx, q in enumerate(QUARTERS):
        print(
            f"Stage {q_idx + 1}/8: {q} — generating assignments...", end=" ", flush=True
        )

        # Generate candidate assignments for this quarter
        candidates = generate_assignments(q, granularity)
        print(f"{len(candidates)} candidates", end=" ", flush=True)

        # Evaluate candidates in parallel
        with Pool(n_workers) as pool:
            results = pool.map(evaluate_assignment, candidates)

        # Filter feasible results
        feasible = [
            (candidates[i][1], r[1], r[2])
            for i, r in enumerate(results)
            if r is not None
        ]
        print(f"→ {len(feasible)} feasible", flush=True)

        if not feasible:
            print(f"  ERROR: No feasible assignments for {q}!")
            return None, None

        # Build new DP table
        new_dp = {}

        for prev_state, (prev_cost, prev_path) in dp.items():
            prev_mt_tuple, prev_tor_tuple = prev_state
            prev_mt = tuple_to_mt(prev_mt_tuple)

            for assignment, new_tor_per_fab_tuple, tor_space_per_fab in feasible:
                # new_tor_per_fab_tuple: ((f1_ws1,...,f1_ws6), (f2_ws1,...), (f3_ws1,...))
                # Compute new TOR tuple (flat)
                new_tor_flat = tuple(
                    new_tor_per_fab_tuple[f_idx][ws_idx]
                    for f_idx in range(len(FABS))
                    for ws_idx in range(len(TOR_WS))
                )

                # Incremental CapEx: sum over all fabs and WS types
                capex = 0
                for f_idx, f in enumerate(FABS):
                    for ws_idx, ws in enumerate(TOR_WS):
                        prev_count = prev_tor_tuple[f_idx * len(TOR_WS) + ws_idx]
                        new_count = new_tor_per_fab_tuple[f_idx][ws_idx]
                        delta = max(0, new_count - prev_count)
                        capex += delta * WS_SPECS[ws][1]

                # Move-out cost
                mo_cost, new_mt = compute_moveout_cost(tor_space_per_fab, prev_mt)

                # Total cost for this transition
                # Q1'26 (q_idx==0) excluded from Excel objective
                transition_cost = capex + mo_cost
                if q_idx == 0:
                    transition_cost = 0
                new_cost = prev_cost + transition_cost

                # New state
                new_mt_tuple = mt_to_tuple(new_mt)
                new_state = (new_mt_tuple, new_tor_flat)

                # Update DP table (keep minimum cost)
                if new_state not in new_dp or new_dp[new_state][0] > new_cost:
                    new_dp[new_state] = (
                        new_cost,
                        prev_path + [(q, assignment, transition_cost)],
                    )

        dp = new_dp
        elapsed = time.time() - start_time
        print(f"  DP states: {len(dp):,}  |  Elapsed: {elapsed:.1f}s")

        # Prune: keep only top-K states by cost to prevent state explosion
        MAX_STATES = 50000
        if len(dp) > MAX_STATES:
            sorted_states = sorted(dp.items(), key=lambda x: x[1][0])
            dp = dict(sorted_states[:MAX_STATES])
            print(f"  Pruned to {MAX_STATES} states")

    # Find best final state
    best_cost = float("inf")
    best_path = None
    for state, (cost, path) in dp.items():
        if cost < best_cost:
            best_cost = cost
            best_path = path

    total_time = time.time() - start_time
    print(f"\nDP completed in {total_time:.1f}s")
    print(f"Best cost found: ${best_cost:,.0f}")

    return best_cost, best_path


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY BASELINE (for comparison)
# ─────────────────────────────────────────────────────────────────────────────


def greedy_cost():
    """Run the greedy solution and return its cost."""
    import subprocess, sys

    result = subprocess.run(
        [sys.executable, "q1b_solution_v2.py"],
        capture_output=True,
        text=True,
    )
    # Parse cost from output
    for line in result.stdout.split("\n"):
        if "Total Cost:" in line:
            cost_str = line.split("$")[1].replace(",", "").strip()
            return int(float(cost_str))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Q1b DYNAMIC PROGRAMMING SOLVER (Parallelized)")
    print("=" * 70)
    print(f"CPUs available: {cpu_count()}")
    print(f"Problem: 8 quarters × 3 nodes × 3 fabs")
    print()

    # ── Phase 1: Coarse search (granularity = 1000 wafers) ──────────────────
    print("PHASE 1: Coarse DP (granularity=1000 wafers)")
    print("-" * 50)
    t0 = time.time()
    best_cost_coarse, best_path_coarse = dp_solve(
        granularity=1000, n_workers=max(1, cpu_count() - 1)
    )
    t1 = time.time()
    print(f"\nPhase 1 completed in {t1 - t0:.1f}s")

    if best_cost_coarse is None:
        print("ERROR: DP found no feasible solution!")
        sys.exit(1)

    print(f"\nCoarse DP best cost: ${best_cost_coarse:,.0f}")

    # ── Phase 2: Fine search (granularity = 500 wafers) ─────────────────────
    print("\nPHASE 2: Fine DP (granularity=500 wafers)")
    print("-" * 50)
    t0 = time.time()
    best_cost_fine, best_path_fine = dp_solve(
        granularity=500, n_workers=max(1, cpu_count() - 1)
    )
    t1 = time.time()
    print(f"\nPhase 2 completed in {t1 - t0:.1f}s")

    if best_cost_fine is None:
        print("ERROR: Fine DP found no feasible solution!")
        best_cost_fine = best_cost_coarse
        best_path_fine = best_path_coarse

    print(f"\nFine DP best cost: ${best_cost_fine:,.0f}")

    # ── Greedy baseline ──────────────────────────────────────────────────────
    GREEDY_COST = 5_207_900_000  # from previous run

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print(f"  Greedy solution:    ${GREEDY_COST:>18,.0f}")
    print(f"  DP (coarse):        ${best_cost_coarse:>18,.0f}")
    print(f"  DP (fine):          ${best_cost_fine:>18,.0f}")

    gap_coarse = (best_cost_coarse - GREEDY_COST) / GREEDY_COST * 100
    gap_fine = (best_cost_fine - GREEDY_COST) / GREEDY_COST * 100
    print(f"\n  DP coarse vs greedy: {gap_coarse:+.3f}%")
    print(f"  DP fine   vs greedy: {gap_fine:+.3f}%")

    if best_cost_fine <= GREEDY_COST:
        print("\n  DP found a BETTER solution than greedy!")
    else:
        print("\n  DP did NOT improve on greedy — greedy is at least as good.")

    # ── Save DP path ─────────────────────────────────────────────────────────
    if best_path_fine:
        print("\n=== DP Best Path ===")
        for q, assignment, cost in best_path_fine:
            print(f"  {q}: cost=${cost:,.0f}")
            for n in NODES:
                f1 = assignment[n][1]
                f2 = assignment[n][2]
                f3 = assignment[n][3]
                print(f"    Node {n}: F1={f1:6d}, F2={f2:6d}, F3={f3:6d}")

    # Save results
    results = {
        "greedy_cost": GREEDY_COST,
        "dp_coarse_cost": best_cost_coarse,
        "dp_fine_cost": best_cost_fine,
        "gap_coarse_pct": gap_coarse,
        "gap_fine_pct": gap_fine,
        "dp_path": [
            {
                "quarter": q,
                "assignment": {
                    str(n): {str(f): v for f, v in fv.items()}
                    for n, fv in assignment.items()
                },
                "cost": cost,
            }
            for q, assignment, cost in (best_path_fine or [])
        ],
    }
    with open("results/dp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to dp_results.json")
