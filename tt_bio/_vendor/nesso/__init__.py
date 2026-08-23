"""Upstream Nesso-1 host pipeline, vendored verbatim (Apache-2.0).

Only the host data path lives here: YAML parsing, tokenization, the featurizer
and the pocket crop. The model itself is not vendored -- 37 of its definitions
are byte-identical to code already in ``tt_bio.boltz2``/``tt_bio.reference``, so
``tt_bio.nesso1`` builds it from those instead of carrying a second copy.

The featurizer is the exception that earns a copy: measured against
``tt_bio/data/featurizer.py``, ``process_atom_features`` scores 0.207 and
``process_token_features`` 0.278 similarity. The two boltz forks have genuinely
diverged, and this checkpoint wants a 390-dim atom feature vector where ours
builds 128. Merging them would risk silent feature drift, so the upstream code
is kept unmodified except for import paths.

Source: github.com/recursionpharma/nesso @ f0156e9, LICENSE alongside.
"""
