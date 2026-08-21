import logging
import os

logger = logging.getLogger("preprocess")
_log_level = os.environ.get("PREPROCESS_LOG_LEVEL", "WARNING").upper()
logger.setLevel(_log_level)
