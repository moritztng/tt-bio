"""Upstream `opendde pred`, unchanged except the trunk chunk size (64): upstream runs <=1024 tokens unchunked, and
chunking changes memory, not the result."""
import sys
from opendde.model import opendde as M
M.OpenDDE._get_dynamic_chunk_size = lambda self, n: 64
from runner.cli import opendde_cli
sys.argv[0] = "opendde"
sys.exit(opendde_cli())
