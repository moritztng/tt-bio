"""Apply the biotite 1.2 compat shim to a vendored atomworks 3.0.4 tree. Usage: shim.py <repo_root>"""
import sys

root = sys.argv[1] + "/tt_bio/_vendor/atomworks/"
p = root + "__init__.py"; s = open(p).read()
first = s[s.index("# Hard-enforce biotite version compatibility"):s.index("del _biotite_check\n") + len("del _biotite_check\n")]
s = s.replace(first, "")
old = '''_REQUIRED_BIOTITE_VERSION = "1.6.0"
if biotite.__version__ != _REQUIRED_BIOTITE_VERSION:
    raise RuntimeError(
        f"atomworks requires biotite=={_REQUIRED_BIOTITE_VERSION}, "
        f"but found biotite=={biotite.__version__}. "
        f"Install the correct version: pip install biotite=={_REQUIRED_BIOTITE_VERSION}"
    )'''
new = '''# Upstream pins biotite 1.6.0, which needs Python >= 3.11. On Python 3.10 the newest biotite is
# 1.2.0; biotite_patch backports the three 1.6 behaviours atomworks relies on that 1.2.0 lacks.
_SUPPORTED_BIOTITE_VERSIONS = ("1.2.0", "1.6.0")
if biotite.__version__ not in _SUPPORTED_BIOTITE_VERSIONS:
    raise RuntimeError(
        f"atomworks supports biotite {' or '.join(_SUPPORTED_BIOTITE_VERSIONS)}, "
        f"but found biotite=={biotite.__version__}."
    )'''
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s)

p = root + "biotite_patch.py"; s = open(p).read()
anchor = "def _update_set_inter_residue_bonds() -> None:"
backport = '''@apply_if_version_lt(biotite.__version__, "1.3.0")
def _backport_biotite_1_3() -> None:
    """Give biotite 1.2.0 (the newest on Python 3.10) the three 1.3+ behaviours atomworks uses:
    ``get_all_residue_positions``, ``BondList.convert_bond_type``, and ``coord()`` accepting
    AtomArray subclasses such as ``AtomArrayPlus`` (1.2 checks the exact type)."""
    import sys

    def get_all_residue_positions(array):
        starts = struc.residues.get_residue_starts(array, add_exclusive_stop=True)
        return np.repeat(np.arange(len(starts) - 1), np.diff(starts))

    def convert_bond_type(self, original_bond_type, new_bond_type):
        mask = self._bonds[:, 2] == original_bond_type
        self._bonds[mask, 2] = new_bond_type

    original_coord = struc.atoms.coord

    def coord(item):
        if isinstance(item, (Atom, AtomArray, AtomArrayStack)):
            return item.coord
        return original_coord(item)

    struc.get_all_residue_positions = get_all_residue_positions
    struc.BondList.convert_bond_type = convert_bond_type
    # every biotite module holding the 1.2 function by reference (celllist binds it as to_coord)
    for name, module in list(sys.modules.items()):
        if name.startswith("biotite."):
            for attr, value in list(vars(module).items()):
                if value is original_coord:
                    setattr(module, attr, coord)


'''
assert s.count(anchor) == 1; s = s.replace(anchor, backport + anchor)
s = s.replace("import biotite.structure as struc\n", "import biotite\nimport biotite.structure as struc\n", 1)
old = "    _update_get_residue_starts()\n    _update_array()"
assert s.count(old) == 1
s = s.replace(old, "    _update_get_residue_starts()\n    _backport_biotite_1_3()\n    _update_array()")
open(p, "w").write(s)
print("patched")
