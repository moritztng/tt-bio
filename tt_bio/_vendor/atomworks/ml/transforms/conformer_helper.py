"""tt-bio: serve `rdkit_utils._conformers_here` over pickles on stdin/stdout (see `_conformers_in_helper`)."""
import pickle
import sys

from rdkit import Chem

from tt_bio._vendor.atomworks.ml.transforms.rdkit_utils import _conformers_here


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # nothing but replies on the pipe
    while True:
        try:
            props, jobs, seeds, timeout, kwargs = pickle.load(inp)
        except EOFError:
            return
        Chem.SetDefaultPickleProperties(props)
        try:
            reply = (True, _conformers_here(jobs, seeds, timeout, kwargs))
        except Exception as e:  # noqa: BLE001 -- raised again in the caller
            reply = (False, e)
        pickle.dump(reply, out)
        out.flush()


if __name__ == "__main__":
    main()
