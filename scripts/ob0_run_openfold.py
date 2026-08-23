"""Console-script stand-in for an openfold3 SOURCE CLONE.

The OpenBind release (v0.5.0) is not on PyPI, so there is no `run_openfold` entry
point to call. Point PYTHONPATH at the clone and run this instead:

    PYTHONPATH=/home/ttuser/ob0_upstream:/home/ttuser/ob0_refdeps \\
        <refenv>/bin/python scripts/ob0_run_openfold.py predict --query-json ...
"""
from openfold3.run_openfold import cli

if __name__ == "__main__":
    cli()
