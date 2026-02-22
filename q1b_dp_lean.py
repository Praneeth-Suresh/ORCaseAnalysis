"""
Script: Q1b Lean DP Solver (Parallelized, TOR-state only)
==========================================================
Key insight: the move-out schedule is DETERMINISTIC given the TOR
tool requirements. The minimum-cost move-out strategy is always to
move out the largest-footprint tools first, and only as many as
needed. Therefore, the MT tool inventory is NOT an independent state
variable — it is fully determined by the history of TOR requirements.

This allows us to reduce the DP state to just the TOR tool inventory,
which is much smaller and allows exact DP with parallelization.

STATE REPRESENTATION
---------------------
State at end of quarter q:
  tor_state = tuple of TOR tool counts for each (fab, ws_type)
  = (t_{f1,A+}, t_{f1,B+}, ..., t_{f1,F+},
     t_{f2,A+}, ..., t_{f2,F+},
     t_{f3,A+}, ..., t_{f3,F+})
  = 18 integers

The MT tool inventory is reconstructed from the initial state and
the history of TOR requirements (deterministic greedy move-out).

DECISION AT EACH STAGE
------------------------
Decision: node-fab assignment w_{q,n,f} for all n, f.
This determines the TOR requirements, which determines:
  - CapEx (incremental TOR purchases)
  - Move-out cost (deterministic from TOR space needs)

PARALLELIZATION
----------------
For each stage, all candidate assignments are evaluated in parallel.
The DP table is pruned to MAX_STATES states per stage to control
memory and runtime.
"""

import json
import math
import time
from multiprocessing import Pool, cpu_count

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

with open("parameters/params.json") as f:
    P = json.load(f)

QUARTERS = P["quarters"]
FABS = [1, 2, 3]
NODES = [1, 2, 3]
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
FAB_SPACE = {f: FAB_SPECS[f]["space"] for f in FABS}
INITIAL_MT = {ws: {f: FAB_SPECS[f]["tools"][ws] for f in FABS} for ws in MINTECH_WS}

# Precompute: TOR tools needed per wafer per node (continuous)
# tor_per_wafer[n][ws_tor] = rpt_tor / (MIN_PER_WEEK * util)
TOR_PER_WAFER = {}
for n in NODES:
    TOR_PER_WAFER[n] = {ws: 0.0 for ws in TOR_WS}
    for _, ws_mt, rpt_mt, ws_tor, rpt_tor in PROCESS_STEPS[n]:
        _, _, util = WS_SPECS[ws_tor]
        TOR_PER_WAFER[n][ws_tor] += rpt_tor / (MIN_PER_WEEK * util)

# Precompute: space per TOR tool
TOR_SPACE = {ws: WS_SPECS[ws][0] for ws in TOR_WS}
TOR_CAPEX = {ws: WS_SPECS[ws][1] for ws in TOR_WS}

# Precompute: MT tool space
MT_SPACE = {ws: WS_SPECS[ws][0] for ws in MINTECH_WS}


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def compute_tor_for_fab(flow_nf):
    """
    flow_nf: {n: wafers} for a single fab.
    Returns (space_used, tor_counts_tuple).
    """
    tor_cont = {ws: 0.0 for ws in TOR_WS}
    for n, wafers in flow_nf.items():
        if wafers > 0:
            for ws in TOR_WS:
                tor_cont[ws] += wafers * TOR_PER_WAFER[n][ws]
    tor_int = tuple(math.ceil(tor_cont[ws]) for ws in TOR_WS)
    space = sum(TOR_SPACE[ws] * tor_int[i] for i, ws in enumerate(TOR_WS))
    return space, tor_int


def compute_moveout_cost_from_mt(tor_space_per_fab, mt_counts):
    """
    Given TOR space needed per fab and current MT tool counts,
    compute the minimum move-out cost (greedy: largest footprint first).
    Returns (total_moveout_cost, new_mt_counts_tuple).
    """
    total_cost = 0
    new_mt = {ws: {f: mt_counts[ws][f] for f in FABS} for ws in MINTECH_WS}

    for f in FABS:
        space_mt = sum(MT_SPACE[ws] * new_mt[ws][f] for ws in MINTECH_WS)
        excess = tor_space_per_fab[f] + space_mt - FAB_SPACE[f]
        if excess > 0.001:
            ws_sorted = sorted(
                [(ws, new_mt[ws][f]) for ws in MINTECH_WS if new_mt[ws][f] > 0],
                key=lambda x: -MT_SPACE[x[0]],
            )
            for ws, count in ws_sorted:
                if excess <= 0.001:
                    break
                sp = MT_SPACE[ws]
                to_move = min(count, math.ceil(excess / sp))
                new_mt[ws][f] -= to_move
                total_cost += to_move * MOVEOUT_COST
                excess -= to_move * sp

    new_mt_tuple = tuple(new_mt[ws][f] for ws in MINTECH_WS for f in FABS)
    return total_cost, new_mt_tuple


def mt_tuple_to_dict(t):
    mt = {}
    idx = 0
    for ws in MINTECH_WS:
        mt[ws] = {}
        for f in FABS:
            mt[ws][f] = t[idx]
            idx += 1
    return mt


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def generate_candidates(q, granularity):
    """
    Generate all feasible (assignment, tor_per_fab, tor_space_per_fab) for quarter q.

    Parameterization:
      - w1_f1: Node 1 in Fab 1 (rest to Fab 2; none to Fab 3)
      - w2_f2: Node 2 in Fab 2 (rest to Fab 1; none to Fab 3)
      - n3_f3: Node 3 in Fab 3 (fixed at max feasible)
      - w3_f1: Node 3 overflow in Fab 1 (rest to Fab 2)

    Node 1 and Node 2 are NOT assigned to Fab 3 because Fab 3 only has
    C workstations (Node 3 uses C; Nodes 1 and 2 do not use C).
    """
    D1 = LOADING[1][q]
    D2 = LOADING[2][q]
    D3 = LOADING[3][q]

    # Max Node 3 in Fab 3
    n3_f3 = 0
    for w in range(0, D3 + 1, granularity):
        sp, _ = compute_tor_for_fab({3: w})
        if sp <= FAB_SPACE[3]:
            n3_f3 = w
        else:
            break
    n3_f3 = min(n3_f3, D3)
    n3_rem = D3 - n3_f3

    candidates = []

    w1_f1_vals = list(range(0, D1 + 1, granularity))
    if D1 not in w1_f1_vals:
        w1_f1_vals.append(D1)

    w2_f2_vals = list(range(0, D2 + 1, granularity))
    if D2 not in w2_f2_vals:
        w2_f2_vals.append(D2)

    w3_f1_vals = list(range(0, n3_rem + 1, granularity))
    if n3_rem not in w3_f1_vals:
        w3_f1_vals.append(n3_rem)

    for w1_f1 in w1_f1_vals:
        w1_f2 = D1 - w1_f1
        for w2_f2 in w2_f2_vals:
            w2_f1 = D2 - w2_f2
            for w3_f1 in w3_f1_vals:
                w3_f2 = n3_rem - w3_f1

                # Compute TOR tools per fab
                sp1, tor1 = compute_tor_for_fab({1: w1_f1, 2: w2_f1, 3: w3_f1})
                sp2, tor2 = compute_tor_for_fab({1: w1_f2, 2: w2_f2, 3: w3_f2})
                sp3, tor3 = compute_tor_for_fab({3: n3_f3})

                # Space feasibility (with no MT tools — worst case)
                if sp1 > FAB_SPACE[1] or sp2 > FAB_SPACE[2] or sp3 > FAB_SPACE[3]:
                    continue

                assignment = {
                    1: {1: w1_f1, 2: w1_f2, 3: 0},
                    2: {1: w2_f1, 2: w2_f2, 3: 0},
                    3: {1: w3_f1, 2: w3_f2, 3: n3_f3},
                }
                tor_per_fab = (tor1, tor2, tor3)  # tuple of tuples
                tor_space = {1: sp1, 2: sp2, 3: sp3}
                candidates.append((assignment, tor_per_fab, tor_space))

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_transition(args):
    """
    Evaluate the cost of transitioning from prev_state to a new state
    via a given assignment.

    args = (prev_tor_state, prev_mt_tuple, assignment, tor_per_fab, tor_space)
    Returns (new_tor_state, new_mt_tuple, transition_cost) or None if infeasible.
    """
    prev_tor_state, prev_mt_tuple, assignment, tor_per_fab, tor_space = args

    # Compute incremental CapEx
    capex = 0
    new_tor_flat = []
    for f_idx, f in enumerate(FABS):
        for ws_idx, ws in enumerate(TOR_WS):
            prev_count = prev_tor_state[f_idx * len(TOR_WS) + ws_idx]
            new_count = tor_per_fab[f_idx][ws_idx]
            delta = max(0, new_count - prev_count)
            capex += delta * TOR_CAPEX[ws]
            new_tor_flat.append(new_count)

    new_tor_state = tuple(new_tor_flat)

    # Compute move-out cost
    prev_mt = mt_tuple_to_dict(prev_mt_tuple)
    mo_cost, new_mt_tuple = compute_moveout_cost_from_mt(tor_space, prev_mt)

    transition_cost = capex + mo_cost
    return (new_tor_state, new_mt_tuple, transition_cost)


# ─────────────────────────────────────────────────────────────────────────────
# DP SOLVER
# ─────────────────────────────────────────────────────────────────────────────


def dp_solve(granularity=500, max_states=20000, n_workers=None):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(
        f"\nLean DP: granularity={granularity}, max_states={max_states}, workers={n_workers}"
    )

    # Initial state
    zero_tor = tuple(0 for _ in range(len(FABS) * len(TOR_WS)))
    initial_mt_tuple = tuple(INITIAL_MT[ws][f] for ws in MINTECH_WS for f in FABS)

    # DP table: {(tor_state, mt_tuple): (cost, path)}
    dp = {(zero_tor, initial_mt_tuple): (0.0, [])}

    start = time.time()

    for q_idx, q in enumerate(QUARTERS):
        print(f"\nStage {q_idx + 1}/8: {q}")

        # Generate candidates
        t0 = time.time()
        candidates = generate_candidates(q, granularity)
        print(f"  Candidates: {len(candidates)} (generated in {time.time() - t0:.2f}s)")

        if not candidates:
            print(f"  ERROR: No feasible candidates!")
            return None, None

        # Build transition tasks
        tasks = []
        prev_states_list = list(dp.items())
        for (prev_tor, prev_mt), (prev_cost, prev_path) in prev_states_list:
            for assignment, tor_per_fab, tor_space in candidates:
                tasks.append((prev_tor, prev_mt, assignment, tor_per_fab, tor_space))

        print(f"  Transitions to evaluate: {len(tasks):,}")

        # Evaluate in parallel
        t0 = time.time()
        chunk = max(1, len(tasks) // (n_workers * 4))
        with Pool(n_workers) as pool:
            results = pool.map(evaluate_transition, tasks, chunksize=chunk)
        print(f"  Evaluated in {time.time() - t0:.2f}s")

        # Build new DP table
        new_dp = {}
        task_idx = 0
        for (prev_tor, prev_mt), (prev_cost, prev_path) in prev_states_list:
            for c_idx, (assignment, tor_per_fab, tor_space) in enumerate(candidates):
                result = results[task_idx]
                task_idx += 1

                if result is None:
                    continue

                new_tor_state, new_mt_tuple, transition_cost = result
                new_cost = prev_cost + transition_cost
                new_state = (new_tor_state, new_mt_tuple)

                if new_state not in new_dp or new_dp[new_state][0] > new_cost:
                    new_dp[new_state] = (
                        new_cost,
                        prev_path + [(q, assignment, transition_cost)],
                    )

        # Prune to max_states
        if len(new_dp) > max_states:
            sorted_states = sorted(new_dp.items(), key=lambda x: x[1][0])
            new_dp = dict(sorted_states[:max_states])
            print(f"  Pruned: {len(new_dp):,} states kept")
        else:
            print(f"  States: {len(new_dp):,}")

        dp = new_dp
        print(f"  Stage elapsed: {time.time() - start:.1f}s total")

    # Best final state
    best_cost = float("inf")
    best_path = None
    for state, (cost, path) in dp.items():
        if cost < best_cost:
            best_cost = cost
            best_path = path

    print(f"\nDP completed in {time.time() - start:.1f}s")
    print(f"Best cost: ${best_cost:,.0f}")
    return best_cost, best_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    GREEDY_COST = 5_207_900_000

    print("=" * 70)
    print("Q1b LEAN DP SOLVER (Parallelized, TOR-state only)")
    print("=" * 70)
    print(f"CPUs: {cpu_count()}")

    results_all = {}

    for gran, max_st in [(2000, 5000), (1000, 10000), (500, 20000)]:
        print(f"\n{'=' * 70}")
        print(f"RUN: granularity={gran}, max_states={max_st}")
        print(f"{'=' * 70}")
        t0 = time.time()
        cost, path = dp_solve(
            granularity=gran, max_states=max_st, n_workers=max(1, cpu_count() - 1)
        )
        elapsed = time.time() - t0

        if cost is None:
            print("No feasible solution found.")
            continue

        gap = (cost - GREEDY_COST) / GREEDY_COST * 100
        print(f"\nResult: cost=${cost:,.0f}, gap={gap:+.4f}%, time={elapsed:.1f}s")
        results_all[f"gran{gran}"] = {"cost": cost, "gap_pct": gap, "time_s": elapsed}

        if path:
            print("\nBest path:")
            for q, assignment, c in path:
                print(f"  {q}: ${c:,.0f}")
                for n in NODES:
                    print(
                        f"    Node {n}: F1={assignment[n][1]:6d}, F2={assignment[n][2]:6d}, F3={assignment[n][3]:6d}"
                    )

        if elapsed > 180:
            print("Time limit reached, stopping further refinement.")
            break

    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"  Greedy solution: ${GREEDY_COST:>18,.0f}")
    for run, r in results_all.items():
        print(
            f"  DP ({run}):  ${r['cost']:>18,.0f}  gap={r['gap_pct']:+.4f}%  time={r['time_s']:.1f}s"
        )

    # Save
    with open("results/dp_lean_results.json", "w") as f:
        json.dump({"greedy_cost": GREEDY_COST, "dp_runs": results_all}, f, indent=2)
    print("\nSaved to dp_lean_results.json")
