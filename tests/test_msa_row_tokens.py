"""The numpy MSA row tokeniser returns exactly what the per-character loop does."""
import random

import numpy as np

from tt_bio.data.parse import _row_tokens, _row_tokens_py


def test_row_tokens_match_loop():
    random.seed(0)
    alpha = "ACDEFGHIKLMNPQRSTVWYXBZUO-acdefgxyz.*0 \t" + "éÄß"
    rows = ["", "ACDEFGHIKL", "aaACDEFGHIKLbb", "-acD-eF-", "xyz", "é-ACéD", "a" * 200 + "A"]
    rows += ["".join(random.choice(alpha) for _ in range(random.randint(1, 80))) for _ in range(2000)]
    for row in rows:
        res, dels = _row_tokens(row)
        want_res, want_dels = _row_tokens_py(row)
        assert np.asarray(res, dtype=np.int64).tolist() == want_res, row
        assert dels == want_dels, row
