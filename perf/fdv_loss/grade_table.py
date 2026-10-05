"""Grade a spread of loss terms against float64 central differences, one row each.

Reproduces the table in docs/bindcraft2.md "Grade it before you spend a campaign on it" from a
pip-installed tt-bio, so the numbers a reader is given are the numbers their own install prints.
Terms are named on the command line; with none, it grades the ones the docs quote.

    PYTHONPATH=<bindcraft2>:<repo>/perf/fdx_loss python3 grade_table.py
"""
import argparse

DEFAULT = ("iptm_loss", "compactness", "plddt_loss", "target_rmsd", "sequence_entropy",
           "interface_pae", "distogram_cce")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=list(DEFAULT))
    arguments = parser.parse_args()
    from tt_bio import bindcraft2
    import effect_terms
    from bindcraft.loss import REGISTERED_LOSSES

    extra = {"aromatic_binder (the term this row adds)": effect_terms.aromatic_binder,
             "resolved_binder (the A/B term)": effect_terms.resolved_binder}
    graded = {name: REGISTERED_LOSSES[name] for name in arguments.names}
    header = "{:46s} {:9s} {:>10s}".format("term", "forward", "worst rel")
    print(header)
    for label, function in list(extra.items()) + list(graded.items()):
        _, worst, dtype = bindcraft2.check_gradient(function)
        print("{:46s} {:9s} {:10.2e}".format(label, str(dtype), worst), flush=True)


if __name__ == "__main__":
    main()
