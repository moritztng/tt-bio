"""
atomworks.ml - Machine Learning utilities for biological data.

This subpackage provides datasets, preprocessing, pipelines, and other ML utilities
for biological data, originally from the atomworks.ml package.
"""

import logging
import os
import warnings

warnings.filterwarnings("once", message="All-NaN slice encountered", category=RuntimeWarning)

logger = logging.getLogger("atomworks.ml")
_log_level = os.environ.get("ATOMWORKS_ML_LOG_LEVEL", os.environ.get("ATOMWORKS_LOG_LEVEL", "WARNING")).upper()
logger.setLevel(_log_level)
