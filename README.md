# Operations Research (Micron Case Analysis)

BACC 2026 case analysis. The goal is to decide **how many chip-making tools to buy, move out, and where to send production** across three fabrication plants (fabs) and eight quarters, so that the total cost — capital expenditure (CapEx) on new tools, tool move-out fees, and cross-fab transfer fees — is minimised.

Three algorithmic families were explored to solve this optimisation problem: **Mixed Integer Linear Programming**, **Greedy Optimisation**, and **Dynamic Programming**. Each family is documented below with its files.

---

## Setup

- Python 3.13+ with a virtual environment recommended.
- Install dependencies:
  ```
  pip install pulp openpyxl
  ```
- All problem data is read directly from the Excel answer sheet **`5)BACC2026ParticipantAnswerSheet.xlsx`** at runtime (no separate data prep step needed for the main solvers).
- Solver outputs are written as JSON files into the [results/](results/) folder.

---

## Background: The Problem

Micron operates **three fabrication plants (Fabs 1, 2, 3)** and manufactures chips on **three technology nodes (Node 1, 2, 3)**.

Each node follows a fixed sequence of **production steps**, where each step consumes time on a specific type of **workstation (WS)**. There are six legacy _Mintech_ workstation types (A–F) and six newer _TOR_ types (A+–F+). TOR tools are faster and take up less floor space per unit of output.

The planning horizon covers **8 quarters (Q1'26 – Q4'27)**. At the start of Q1'26 the tool inventory is fixed. From Q2'26 onwards the model can:

- **Buy** new TOR tools (CapEx).
- **Move out** Mintech tools at \$1M per tool.
- **Transfer** wafers between fabs mid-process at \$650 per wafer per transfer event.

The key constraint is **floor space**: each fab has a maximum m² capacity, so you cannot simply keep buying tools forever.

The **tool requirement formula** ties everything together. For a workstation of type `ws` in a given fab, the number of tools needed equals:

$$\text{tools required} = \sum_{\text{steps using } ws} \frac{\text{wafers processed} \times \text{RPT (min)}}{10080 \times \text{utilisation}}$$

where RPT is the processing time per wafer and 10 080 is the number of minutes in a week. If this value is, say, 4.7, you need to buy ⌈4.7⌉ = 5 physical tools.

---

## Approach 1 — Mixed Integer Linear Programming (MILP)

### What is MILP?

MILP is a mathematical optimisation technique. You define:

- **Decision variables** — the quantities you want to choose (e.g. how many tools to buy).
- **Constraints** — rules the solution must obey (e.g. "total floor space used ≤ fab capacity").
- **Objective function** — the number you want to minimise (e.g. total cost).

A computer solver (here: CBC via the PuLP library) then searches through all possible combinations of variable values to find the cheapest feasible solution. The _integer_ part means some variables must be whole numbers (you cannot buy 2.7 tools).

### Files

---

#### [`q1b_correct_solver.py`](q1b_correct_solver.py) — Step-Level Loading MILP _(primary solver)_

**What makes this different from the other MILP files:** This is the most complete and correct formulation. It tracks the production loading _per individual process step_, not just per-node totals, which makes the tool-requirement and cross-fab transfer calculations exact.

**How it works, step by step:**

1. **Data loading.** All parameters (demand, steps, workstation specs, initial tool counts, fab space) are read from the Excel answer sheet at runtime via `load_data_from_excel()`.

2. **Quarter-by-quarter solve.** The model solves one quarter at a time, passing the tool inventory from the end of one quarter as the starting point for the next.

3. **Decision variables per quarter:**
   - `L_mt[node, step, fab]` — integer number of wafers of a given node, at a given process step, processed on _Mintech_ tools in a given fab.
   - `L_tor[node, step, fab]` — same for _TOR_ tools (only available from Q2'26).
   - `T[ws][fab]` — integer tool count of type `ws` in `fab` at the end of the quarter.
   - `P[ws][fab]` — tools **purchased** this quarter.
   - `MO[ws][fab]` — tools **moved out** this quarter.
   - `dm / dp[node, step, fab]` — auxiliary variables for counting cross-fab wafer transfers.

4. **Constraints:**
   - _Demand:_ For every node and every step, the sum of loading across all three fabs must equal that quarter's demand.
   - _Tool capacity:_ $\sum \frac{L \times RPT}{10080 \times u} \leq T_{ws,fab}$ for each workstation in each fab.
   - _Inventory balance:_ $T_{new} = T_{old} + P - MO$.
   - _Move-out limit:_ $MO \leq T_{old}$ (cannot move out tools you do not have).
   - _Space:_ $\sum_{ws} T_{ws,fab} \times \text{space}_{ws} \leq \text{fab capacity}$.
   - _Transfer delta:_ $dp - dm = L_{step+1,fab} - L_{step,fab}$ — this linearises the absolute-value transfer count so the transfer cost can be included in the objective without using non-linear terms.

5. **Objective:** $\min \; \text{CapEx} + \text{MoveOut cost} + \text{Transfer cost}$, where transfer cost = $\sum dm_{n,s,f} \times \$650$.

6. **Output:** Full tool plan and wafer flow plan saved to `results/q1b_correct_results.json`.

---

#### [`q1b_solver.py`](q1b_solver.py) — Full 8-Quarter MILP

**How it differs:** Builds a **single giant MILP model** covering all 8 quarters simultaneously, rather than solving one quarter at a time. This gives the solver more global freedom (it can, for example, decide to over-buy tools in Q2'26 to avoid higher space costs later) but makes the problem much harder to solve.

**How it works:**

1. Loads parameters from `parameters/params.json`.
2. Creates variables for every combination of (quarter, node, step, fab, tool type) — resulting in tens of thousands of variables.
3. Links quarters together via inventory-balance constraints so that tools bought in one quarter persist into the next.
4. Minimises the sum of CapEx + move-out cost + transfer cost across all 8 quarters jointly.
5. The CBC solver is given a 5-minute time limit; due to the size of the problem it typically finishes at around an 11% optimality gap rather than the proven global optimum.

**Key limitation:** The sheer number of integer variables (all tool counts, all step-level flows, all transfer deltas across all 8 quarters) makes this the hardest formulation to solve to optimality.

---

#### [`q1b_heuristic_solver.py`](q1b_heuristic_solver.py) — Heuristic-Guided MILP

**How it differs:** Reduces the search space by **fixing the node-to-fab assignment upfront** using domain knowledge, then solves only the tool-count decisions as a (much smaller) LP.

**How it works:**

1. **Heuristic phase:** Assign each node to fabs based on which workstations each fab already has:
   - Node 1 (uses A, D, F steps) → Fab 1 (which starts with A=50, D=50, F=90 tools).
   - Node 2 (uses B, E, F steps) → Fab 2 (which starts with B=30, E=30, F=60 tools).
   - Node 3 (uses C, D, E, F steps) → Fab 3 as primary, overflow to Fab 1/2.
2. **LP phase:** With node-fab assignment fixed, solve a continuous LP to find the minimum-cost tool-purchase and move-out schedule. Tool counts are then rounded up to integers.
3. **Result:** Much faster than the full MILP because the binary node-fab assignment decisions are removed. The trade-off is that the fixed assignment might not be globally optimal.

---

#### [`q1b_final_lp.py`](q1b_final_lp.py) — Definitive LP (All-TOR Strategy)

**Core assumption:** All production is run exclusively on TOR tools. This is justified because TOR tools have higher utilisation _and_ a smaller floor-space footprint per unit of throughput, making them strictly better than Mintech tools once you ignore the CapEx cost.

**How it works:**

1. Builds a continuous LP (no integrality constraints) covering all 8 quarters.
2. Variables are: wafer flow per (quarter, node, fab), TOR tool counts per (quarter, ws, fab), and Mintech tool move-outs.
3. Minimises CapEx (TOR purchases) + move-out cost. Transfer cost is excluded because the model assumes each node stays in one fab per quarter.
4. The LP relaxation is solved; results are rounded up for the final integer solution.

**Note:** This model was found to produce infeasible relaxations when the initial Mintech tool inventory plus required TOR tools exceed individual fab space limits. The model needs aggressive move-outs to be feasible.

---

#### [`q1b_lp_clean.py`](q1b_lp_clean.py) — Clean LP (All-TOR, Refactored)

A cleaner rewrite of `q1b_final_lp.py` with improved code structure and helper functions (`compute_tor_req_per_wafer`, `compute_space_per_wafer`, `total_space_per_wafer`). The mathematical formulation is identical — it solves the same LP model but is easier to read and debug.

---

## Approach 2 — Greedy Optimisation

### What is Greedy Optimisation?

A greedy algorithm makes the **locally best decision at each step** without looking ahead. It is much faster than MILP because it never considers all combinations, but it is not guaranteed to find the global optimum. In this problem, greedy decisions are used to assign nodes to fabs (choosing the best-looking assignment quarter by quarter) and to schedule move-outs (removing tools only when space absolutely requires it, removing the largest tools first to free the most space per dollar spent).

### Files

---

#### [`q1b_optimal_solution.py`](q1b_optimal_solution.py) — Greedy Assignment + Just-In-Time Purchasing

**How it works (four phases):**

1. **Greedy node-fab assignment.** For each quarter, assign nodes to fabs using a _space-per-wafer_ metric:

   $$\text{space per wafer} = \sum_{ws} \frac{\text{RPT}_{ws}}{10080 \times u_{ws}} \times \text{m}^2_{ws}$$

   Starting with Node 1 filling Fab 1, Node 2 filling Fab 2, Node 3 filling Fab 3, the greedy rule shifts overflow to the next available fab when a fab runs out of space.

2. **Compute TOR tool requirements.** Given the assignment, calculate the exact number of TOR tools needed per workstation per fab per quarter using the tool-requirement formula, then ceiling-round to integers.

3. **Just-in-time purchasing.** Buy TOR tools only in the quarter they are first needed:

   $$\text{buy}_{q,ws,f} = \max(0,\ \text{req}_{q,ws,f} - \text{req}_{q-1,ws,f})$$

   This minimises CapEx timing (though without a discount rate it does not change total cost).

4. **Greedy move-out scheduling.** When the space occupied by existing Mintech tools plus the newly required TOR tools exceeds a fab's capacity, move out the **largest-footprint Mintech tools first** until enough space is freed. This greedy rule minimises the number of tools moved out (because removing one large tool frees more space than removing one small tool).

---

#### [`q1b_solution_v2.py`](q1b_solution_v2.py) — Space-Aware Greedy (Rounding Fix)

**Problem it solves:** The greedy assignment in `q1b_optimal_solution.py` computes space usage using _continuous_ tool requirements. When tool counts are rounded up to integers (ceiling), the actual space used can exceed the limit even if the continuous calculation said it would fit.

**Fix applied:** Binary search is used to find the maximum integer number of wafers that fits in each fab _after_ ceiling-rounding the tool counts:

```
find largest w such that ⌈TOR_tools_needed(w)⌉ × space_per_tool ≤ fab_capacity
```

The `compute_tor_space` and `max_wafers_in_fab` helpers implement this, ensuring the final assignment is always physically feasible even after rounding.

---

## Approach 3 — Dynamic Programming (DP)

### What is Dynamic Programming?

Dynamic Programming (DP) solves problems by breaking them into **stages** (here: quarters) and remembering the cheapest way to reach each possible **state** (here: the tool inventory at the end of a quarter). By building up the solution from the first quarter to the last, and only keeping the best path to each state, DP guarantees a globally optimal solution — provided the state space is manageable in size.

The core recurrence is:

$$\text{cost}^*(q, \text{state}) = \min_{\text{decision}} \Big[ \text{cost}(q, \text{decision}) + \text{cost}^*(q+1, \text{next state}) \Big]$$

In plain language: _the cheapest way to reach the end of quarter q in a given state equals the cheapest single-quarter cost plus the cheapest way to finish from there._

### Files

---

#### [`q1b_dp_solver.py`](q1b_dp_solver.py) — Parallelised DP (Full State)

**State representation:** The state is the TOR tool inventory across all fabs — a tuple of 18 integers (6 TOR workstation types × 3 fabs). Mintech tool counts are not part of the state because, given a TOR requirement, the cheapest move-out schedule is deterministic: always move out the largest Mintech tools first.

**How it works:**

1. **Enumerate candidates.** For each quarter, generate all feasible node-fab assignments at a coarse granularity (e.g. every 2,000 wafers). Each candidate is evaluated in parallel using Python's `multiprocessing.Pool`. A candidate is infeasible and discarded if: (a) demand is not met, or (b) the required TOR tools alone exceed a fab's total floor space.

2. **DP table.** For each quarter, maintain a table mapping `state → (best cost so far, path)`. For each candidate assignment, compute the incremental CapEx (cost of buying new TOR tools above what was already owned) and the greedy move-out cost, then update the table:

   $$\text{cost}[\text{new state}] = \min\Big(\text{cost}[\text{new state}],\ \text{cost}[\text{prev state}] + \Delta\text{CapEx} + \text{MoveOut cost}\Big)$$

3. **Pruning.** The state space can still be enormous. The solver caps the number of states kept per quarter (`MAX_STATES`) to control memory and runtime.

4. **Output:** The optimal 8-quarter path (sequence of node-fab assignments) and its total cost.

---

#### [`q1b_dp_lean.py`](q1b_dp_lean.py) — Lean Parallelised DP (Reduced State)

**Key improvement over `q1b_dp_solver.py`:** The observation that Node 3 is the only node that uses Fab 3 (because Fab 3 only has C workstations, which Node 3 uses exclusively) reduces the effective assignment space from a 3×3 matrix to just four free variables: how much of Node 1 goes to Fab 1 (the rest to Fab 2), how much of Node 2 goes to Fab 2 (the rest to Fab 1), how much of Node 3 fills Fab 3 (the rest splits between Fab 1 and Fab 2), and how much of that remainder goes to Fab 1.

**How it works:**

1. **Precomputation.** Before the DP starts, the TOR tools needed _per wafer per node_ are computed once:
   $$\text{tor\_per\_wafer}[n][ws] = \sum_{\text{steps of node } n \text{ using } ws} \frac{RPT_{ws}}{10080 \times u_{ws}}$$

2. **Candidate generation.** For each quarter, loop over a grid of the four free variables (at adjustable granularity). For each grid point: compute TOR space per fab, discard if any fab overflows, otherwise record the candidate.

3. **Parallel evaluation.** All candidates are evaluated in parallel. Each worker returns the TOR tool requirements (as a hashable tuple for the DP table key) and whether the assignment is feasible.

4. **DP update.** Same recurrence as `q1b_dp_solver.py`, but with a much smaller state space and finer granularity possible within the same runtime budget.

5. **Greedy move-out sub-routine.** For each candidate, the Mintech move-out cost is computed greedily: if the TOR tools for a fab need more space than remains after Mintech tools, remove the largest-footprint Mintech tools first until enough space is freed.

6. **Output:** Best path saved to `results/dp_lean_results.json` alongside total cost, CapEx breakdown, and move-out breakdown.

---

#### [`q1b_dp_refine.py`](q1b_dp_refine.py) — DP Refinement via Local Neighbourhood Search

**Purpose:** The DP solvers use a coarse grid (e.g. every 1,000 or 2,000 wafers) to keep runtime manageable, which means the optimal solution may be slightly off the grid. This file _refines_ the best path found by `q1b_dp_lean.py` using two techniques:

1. **Local neighbourhood search.** Starting from the best-known 8-quarter path, try perturbing each quarter's assignment by ±`radius` wafers at a finer `granularity` (e.g. 100 wafers). Accept any improvement and repeat until no improvement is found. This is a classic _iterative improvement_ heuristic:

   $$\text{while improvement exists:}\quad \text{for each quarter, try nearby assignments and accept if cheaper}$$

2. **Beam search.** Run the DP forward at medium granularity (500 wafers) but keep only the top-`B` states at each stage (the "beam"). This is a memory-efficient approximate DP that validates whether the neighbourhood search result is consistent with a broader exploration of the state space.

**Output:** Refined path and cost saved to `results/dp_refine_results.json`.

---

## Files Marked for Deletion

The following file does **not** implement any optimisation algorithm. It is a post-processing script that takes a hardcoded best-known DP path (embedded directly in the source) and reconstructs a detailed JSON output from it. Because the path is hardcoded, any change to the problem data or solver would require manually updating this file, making it redundant and a potential source of stale results:

| File                                             | Reason                                                                                                                                              |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`q1b_build_solution.py`](q1b_build_solution.py) | Pure post-processing: reconstructs solution JSON from a hardcoded assignment — no optimisation logic. Superseded by the output of `q1b_dp_lean.py`. |

---

## Summary Table

| File                                               | Category   | Key Idea                                                 |
| -------------------------------------------------- | ---------- | -------------------------------------------------------- |
| [q1b_correct_solver.py](q1b_correct_solver.py)     | MILP       | Step-level loading, quarter-by-quarter, reads from Excel |
| [q1b_solver.py](q1b_solver.py)                     | MILP       | Full 8-quarter simultaneous MILP                         |
| [q1b_heuristic_solver.py](q1b_heuristic_solver.py) | MILP       | Heuristic node-fab assignment, then LP for tools         |
| [q1b_final_lp.py](q1b_final_lp.py)                 | MILP       | All-TOR LP, joint 8-quarter model                        |
| [q1b_lp_clean.py](q1b_lp_clean.py)                 | MILP       | Clean rewrite of q1b_final_lp                            |
| [q1b_optimal_solution.py](q1b_optimal_solution.py) | Greedy     | Space-per-wafer assignment + greedy move-outs            |
| [q1b_solution_v2.py](q1b_solution_v2.py)           | Greedy     | Same as above with integer-safe binary-search rounding   |
| [q1b_dp_solver.py](q1b_dp_solver.py)               | DP         | Parallelised multi-stage DP, full TOR-state              |
| [q1b_dp_lean.py](q1b_dp_lean.py)                   | DP         | Lean DP with reduced 4-variable assignment space         |
| [q1b_dp_refine.py](q1b_dp_refine.py)               | DP         | Local neighbourhood search + beam search refinement      |
| [q1b_build_solution.py](q1b_build_solution.py)     | **Delete** | Hardcoded post-processor, no optimisation                |
