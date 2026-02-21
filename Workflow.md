# Micron Chip Production Optimization

### 2026 NUS-ISE Business Analytics Case Competition

## Part A

### Problem Definition

This is a **multi-period, multi-site, multi-product capacity planning** problem. Across 8 quarters (Q1'26–Q4'27), production of 3 chip nodes must be allocated across 3 fabs, with decisions on TOR tool purchasing. Move-outs of existing mintech tools are **not permitted** in Part A (C6 below).

**Decision Variables:**

| Variable       | Symbol              | Type       | Description                                                                 |
| -------------- | ------------------- | ---------- | --------------------------------------------------------------------------- |
| Flow (mintech) | $w^{MT}_{q,n,s,f}$  | Continuous | Wafers/week of node$n$ at step $s$ on mintech tools in fab $f$, quarter $q$ |
| Flow (TOR)     | $w^{TOR}_{q,n,s,f}$ | Continuous | Wafers/week of node$n$ at step $s$ on TOR tools in fab $f$, quarter $q$     |
| TOR tools      | $t_{q,ws,f}$        | Integer    | TOR workstations of type$ws$ in fab $f$ at end of quarter $q$               |
| Mintech tools  | $m_{q,ws,f}$        | Integer    | Mintech workstations of type$ws$ in fab $f$ at end of quarter $q$           |
| TOR purchases  | $b_{q,ws,f}$        | Integer    | TOR tools purchased in quarter$q$                                           |
| Move-outs      | $o_{q,ws,f}$        | Integer    | Mintech tools moved out in quarter$q$                                       |

**Constraints:**

**C1. Demand (equality):**

$$
\sum_{f \in F} \left(w^{MT}_{q,n,s,f} + w^{TOR}_{q,n,s,f}\right) = D_{q,n} \quad \forall q, n, s
$$

**C2. Capacity (mintech tools):**

$$
\sum_{n,s} w^{MT}_{q,n,s,f} \cdot \frac{RPT^{MT}_{n,s}}{T_{week} \cdot u^{MT}_{ws}} \leq m_{q,ws,f} \quad \forall q, ws, f
$$

**C3. Capacity (TOR tools):**

$$
\sum_{n,s} w^{TOR}_{q,n,s,f} \cdot \frac{RPT^{TOR}_{n,s}}{T_{week} \cdot u^{TOR}_{ws}} \leq t_{q,ws,f} \quad \forall q, ws, f
$$

where $RPT^{g}_{n,s}$ is the runtime per wafer for node $n$ at step $s$ on tool generation $g$, $T_{week} = 10{,}080$ min/week, and $u^{g}_{ws}$ is the utilization rate.

**C4. Space:**

$$
\sum_{ws} s_{ws} \cdot m_{q,ws,f} + \sum_{ws} s_{ws^+} \cdot t_{q,ws^+,f} \leq A_f \quad \forall q, f
$$

**C5. Tool inventory dynamics:**

$$
m_{q,ws,f} = m_{q-1,ws,f} - o_{q,ws,f} \quad \forall q, ws, f
$$

$$
t_{q,ws,f} = t_{q-1,ws,f} + b_{q,ws,f} \quad \forall q, ws, f
$$

**C6. No move-outs (Part A restriction):**

$$
o_{q,ws,f} = 0 \quad \forall q, ws, f
$$

**C7. Non-negativity:**

$$
w^{MT}_{q,n,s,f},\ w^{TOR}_{q,n,s,f},\ t_{q,ws,f},\ m_{q,ws,f},\ b_{q,ws,f},\ o_{q,ws,f} \geq 0
$$

**Objective Function:**

$$
\min \sum_{q,ws,f} C^{TOR}_{ws} \cdot b_{q,ws,f}
$$

where $C^{TOR}_{ws}$ is the CapEx per TOR tool. Move-out costs are excluded as move-outs are forbidden.

---

### Data Extraction

The following data were extracted from the case workbook and saved to `params.json` (`data_extractor.py`):

1. Workstation specs: space footprint (m²), CapEx, and utilization rate for each workstation type across both mintech and TOR generations.
2. Process recipes: for each node, the ordered list of process steps with workstation assignments and runtime per wafer (RPT) for both mintech and TOR.
3. Fab specs: floor area ($A_f$) and initial mintech tool inventory for each fab.
4. Demand schedule: quarterly wafer demand $D_{q,n}$ for each node over 8 quarters.

---

### Model Formulation

The Part A problem was formulated as a **Mixed Integer Linear Program (MILP)** with step-level flow split variables (`q1a_solver.py`). Both mintech and TOR tools can process wafers simultaneously on each step, giving the solver flexibility to use existing mintech capacity while purchasing TOR tools only where needed. The model was solved using the CBC branch-and-bound solver via PuLP.

Key modelling choices:

- Flow variables are defined at the **step level** (not node level) to correctly account for per-step tool consumption on heterogeneous tool generations.
- Mintech tool counts are fixed at their initial values due to C6 (no move-outs), so $m_{q,ws,f} = m_{0,ws,f}$ for all $q$.
- TOR purchases are integer-constrained; flow variables are continuous.

---

### Result

The model returned **Infeasible**.

The initial mintech tool inventory occupies 2,869 m² of the 3,500 m² total fab floor area across all three fabs. With move-outs prohibited (C6), only 631 m² remains available for TOR tools. This is insufficient to meet demand in any quarter beyond Q1'26, as Node 3 demand alone grows from 3,000 to 16,000 wafers/week by Q4'27, requiring substantially more capacity.

---

### Validation

**Infeasibility proof via space analysis:**

The space required per wafer/week (using all-TOR tools, the most space-efficient option) was computed as:

$$
\sigma^{TOR}_n = \sum_{ws} s^{TOR}_{ws} \cdot \frac{RPT^{TOR}_{n,ws}}{T_{week} \cdot u^{TOR}_{ws}}
$$

| Node | $\sigma^{MT}_n$ (m²/wfr/wk) | $\sigma^{TOR}_n$ (m²/wfr/wk) |
| ---- | --------------------------- | ---------------------------- |
| 1    | 0.1258                      | 0.1009                       |
| 2    | 0.1161                      | 0.0936                       |
| 3    | 0.2148                      | 0.1487                       |

Even in the best case (100% TOR tools), the space required across all quarters exceeds the 631 m² available for TOR expansion:

| Quarter | Space needed (all-TOR) | TOR headroom | Feasible |
| ------- | ---------------------- | ------------ | -------- |
| Q1'26   | 2,124.7 m²             | 631 m²       | ✗        |
| Q4'27   | 3,283.3 m²             | 631 m²       | ✗        |

**Conclusion:** Part A is infeasible at the given demand levels. The no-move-out constraint (C6) is the binding restriction.

---

## Part B

### Problem Definition

Part B relaxes constraint C6 from Part A, permitting mintech tool move-outs at a cost of $C^{MO} = \$1{,}000{,}000$ per tool. All other constraints (C1–C5, C7) remain as defined in Part A. The objective function is extended to include move-out costs:

$$
\min \sum_{q,ws,f} C^{TOR}_{ws} \cdot b_{q,ws,f} + C^{MO} \cdot \sum_{q,ws,f} o_{q,ws,f}
$$

---

### Data Extraction

Same data as Part A (`params.json`). No additional extraction was required.

---

### Model Formulation

A full MILP was found to be computationally intractable (CBC timed out after 600 s).

We considered the following alternative approaches:

| Strategy                      | Description                                                                                                                                                                                       | Pros                                                 | Cons                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 1.**Two-Phase Optimization**  | Phase 1: Solve the Part (a) MILP. Phase 2: Identify underutilized tools from the solution and run a second optimization to evaluate move-out decisions.                                           | Simpler than a single large model.                   | Suboptimal, as move-out decisions are not integrated with allocation decisions.     |
| 2.**Cost-Benefit Heuristic**  | Develop a rule-of-thumb: if the cost of moving out a tool is less than the opportunity cost of the space it occupies (e.g., value of installing a more productive TOR tool), then move it out.    | Intuitive and easy to apply.                         | Fails to capture the network effects and complex trade-offs of the system.          |
| 3.**Lagrangian Relaxation**   | Relax the space constraints and add them to the objective function with a penalty (Lagrangian multiplier). This decomposes the problem by fab, making it easier to solve. Iterate on multipliers. | Effective for large, constrained problems.           | Requires careful tuning of multipliers; may not yield a feasible solution directly. |
| 4.**Robust Optimization**     | While demand is deterministic here, this approach could be used to create a tool plan that is robust to small variations in processing times or tool availability.                                | Creates a more resilient plan.                       | Overkill for a deterministic problem; may lead to overly conservative solutions.    |
| 5.**Simulation-Optimization** | Use a discrete-event simulation model of the fab network. An optimization algorithm (like a genetic algorithm) iteratively adjusts the allocation rules to find a low-cost solution.              | Can handle complex, non-linear relationships.        | Computationally expensive; may not find the true global optimum.                    |
| 6.**Greedy Heuristic**        | For each process step, assign wafers to the fab with the lowest immediate marginal cost, based on available capacity and RPT. Purchase tools only when capacity is exhausted.                     | **Simple to implement, fast computation.**           | Complex to implement correctly; convergence can be slow.                            |
| 7.**Dynamic Programming**     | Break the problem down by quarter. Solve for the optimal strategy in the final quarter, then work backward, using the end-state of one quarter as the initial state for the previous one.         | Guarantees optimality for the subproblems it solves. | Can be computationally intensive; requires specialized software.                    |

As a simple algorithm, we chose to use a **greedy decomposition** (`q1b_solution_v2.py`), which exploits the following natural sub-problem structure:

1. **Node-fab assignment** — For each quarter, wafers are assigned to fabs greedily: Node 1 primary to Fab 1 (overflow to Fab 2), Node 2 primary to Fab 2 (overflow to Fab 1), Node 3 primary to Fab 3 (overflow to Fab 1, then Fab 2). A space-aware binary search finds the maximum integer-valued flow for each fab without violating the space constraint (C4).
2. **TOR tool requirement** — Given the flow assignment, the minimum required TOR tools per workstation type per fab are computed by ceiling-rounding the continuous tool requirement:

   $$
   t_{q,ws,f} = \left\lceil \sum_{n,s} w_{q,n,f} \cdot \frac{RPT^{TOR}_{n,s}}{T_{week} \cdot u^{TOR}_{ws}} \right\rceil
   $$

3. **Just-in-time TOR purchasing** — Tools are purchased only when demand increases, minimising the time cost of capital:

   $$
   b_{q,ws,f} = \max\!\left(0,\ t_{q,ws,f} - t_{q-1,ws,f}\right)
   $$

4. **Deferred move-out scheduling** — Mintech tools are moved out only when required to satisfy the space constraint (C4). In each quarter, if $S^{TOR}_{q,f} + S^{MT}_{q,f} > A_f$, mintech tools are removed in order of largest footprint first until the constraint is satisfied.

---

### Result

| Component             | Amount             |
| --------------------- | ------------------ |
| CapEx (TOR purchases) | $4,669,900,000     |
| OpEx (move-outs)      | $538,000,000       |
| **Total Cost**        | **$5,207,900,000** |

538 mintech tools were moved out across the planning horizon, with 273 (51%) occurring in Q1'26 to immediately free capacity for TOR tools. By Q4'27, all production runs on TOR tools and space utilisation reaches 94% across fabs.

---

### Validation

All constraints were verified programmatically against the computed solution:

| Constraint                                       | Status                                |
| ------------------------------------------------ | ------------------------------------- |
| C1. Demand (8 quarters × 3 nodes)                | ✓ All satisfied                       |
| C4. Space (8 quarters × 3 fabs)                  | ✓ All satisfied (87–100% utilisation) |
| C2/C3. Capacity (8 quarters × 6 WS × 3 fabs)     | ✓ All satisfied                       |
| C5/C7. Tool non-negativity and inventory balance | ✓ All satisfied                       |

The solution was then written to the answer spreadsheet (`fill_excel.py`) and all built-in validation cells confirmed TRUE.
