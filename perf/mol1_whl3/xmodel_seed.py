"""xmodel_ab.py at another diffusion seed: python xmodel_seed.py SEED <xmodel_ab args...> (tt_baseline.SEED set first)."""
import runpy, sys
from pathlib import Path
seed, args = int(sys.argv[1]), sys.argv[2:]
tree = Path(args[args.index("--tree") + 1]).resolve()
sys.path.insert(0, str(tree)); sys.path.insert(0, str(tree / "scripts" / "gpu_vs_tt"))
import tt_baseline
tt_baseline.SEED = seed
sys.argv = [str(Path(__file__).with_name("xmodel_ab.py"))] + args
runpy.run_path(sys.argv[0], run_name="__main__")
