import sys
import bindcraft, jax, torch, tt_bio, ttnn
from tt_bio import duotraj
print("python", sys.version.split()[0], "jax", jax.__version__, "torch", torch.__version__)
print("tt_bio from", tt_bio.__file__)
print("bindcraft from", bindcraft.__file__)
print("auto_trajectories(544)", duotraj.auto_trajectories(544))
print("auto_trajectories(288)", duotraj.auto_trajectories(288))
