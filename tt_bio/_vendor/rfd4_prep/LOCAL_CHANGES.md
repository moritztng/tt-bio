# Local changes to the vendored RFD4 host package

Upstream: RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f, on atomworks 3.0.4.
Changes for speed keep every output byte for byte: the CIFs `write_cif` writes and the tensors
`prepare_cif` returns.

- `_cif/utils_coordinate_transfer.py`, `copy_generated_coordinates_to_input`: each token's atoms are
  selected by index, built once, instead of a full-length mask per token and atom name.
- `_cif/utils_coordinate_transfer.py`, the same function: a residue whose atom inventory the model
  changed gets its new atoms from a template built once per residue type, atom set and residue
  annotations, then relabelled with the residue's position (`POSITION_ANNOTATIONS`). A template's
  defaults are functions of its names, elements, bonds and type flags, never of its position.
  `get_residue_level_annotations` reads a constant annotation without `np.unique` when every entry
  is bit for bit the first.
- `_cif/transforms_invariants.py`, `check_standard_annotation_level_violations`: a segment is uniform
  when each entry equals its neighbour, checked for all segments at once
  (`_uniform_per_segment`) instead of an `np.unique` per segment.

`tests/test_rfd4_host_exact.py` holds both against upstream's path.
