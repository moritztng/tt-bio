"""Condition system for molecular design and annotation management."""

from tt_bio._vendor.atomworks.ml.conditions.base import CONDITIONS, ConditionBase
from tt_bio._vendor.atomworks.ml.conditions.conditions import *  # noqa: F403

# --- alias table ---
Condition = CONDITIONS
C_SEQ = CONDITIONS.sequence
C_IDX = CONDITIONS.index
C_DIS = CONDITIONS.distance
C_CRD = CONDITIONS.coordinate
C_CHA = CONDITIONS.chain
C_CTR = CONDITIONS.c_terminus
C_NTR = CONDITIONS.n_terminus
# --------------------
