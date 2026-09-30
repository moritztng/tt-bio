"""rung.py with taped_ttnn.SDPA_SCORE_BUDGET set first (B2P_MB), for budget_fp.sh."""
import os, runpy, sys
import tt_bio.taped_ttnn as TT
TT.SDPA_SCORE_BUDGET = int(os.environ["B2P_MB"]) << 20
sys.argv = ["rung.py", "--params", "/home/ttuser/bcx_e2e/af2_params", "--out", os.environ["B2P_OUT"],
            "--target", "hIL2R", "--binder", "100", "--rounds", "2", "--trajectories", "1",
            "--footprint"]
runpy.run_path("perf/bgx_size/rung.py", run_name="__main__")
