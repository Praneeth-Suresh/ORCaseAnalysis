# Operations Research (Micron Case Analysis)

- BACC 2026 case analysis with scripts to extract data, build parameters, and run optimization models for Q1a/Q1b scenarios.
- Core scripts: data extraction ([data_extractor.py](data_extractor.py)), MILP/LP solvers for Q1a/Q1b ([q1a_solver.py](q1a_solver.py), [q1b_solver.py](q1b_solver.py), [q1b_final_lp.py](q1b_final_lp.py), [q1b_lp_clean.py](q1b_lp_clean.py)).
- Inputs are hard-coded from the problem statement; parameter JSON is generated into [parameters/params.json](parameters/params.json). Solver outputs write into the [results](results) folder.
- Current LP formulations (q1b_final_lp, q1b_lp_clean) report infeasible relaxations under existing constraints/space assumptions; see Workflow for details.

## Setup instruction

- Python 3.13+ with a virtual environment recommended.
- Install dependencies in the venv:
  - `pip install pulp openpyxl`
- Generate parameters before running solvers:
  - `python data_extractor.py`
- Ensure the [parameters](parameters) folder exists (created automatically by data_extractor) and writable.

## Workflow

- **Data prep**: Run [data_extractor.py](data_extractor.py) to validate base data and create [parameters/params.json](parameters/params.json).
- **Q1a MILP**: Run [q1a_solver.py](q1a_solver.py) for the no-move-out scenario (mintech + TOR). Expect infeasibility/space violations as per model description.
- **Q1b MILP (full)**: Run [q1b_solver.py](q1b_solver.py) for the move-out-allowed scenario. CBC may require extended time; previous attempts hit 11% gap after ~5 minutes without finalizing a solution.
- **Q1b LP variants**: [q1b_final_lp.py](q1b_final_lp.py) and [q1b_lp_clean.py](q1b_lp_clean.py) currently return infeasible linear relaxations (space violations in Fab 1/2/3 for multiple quarters). No better solution is available without relaxing constraints or altering the model (e.g., redistributing flow, permitting more aggressive move-outs, or changing space limits).
- **Outputs**: Solvers write JSON reports into [results](results). Review the console for feasibility status and constraint violations when runs complete.
