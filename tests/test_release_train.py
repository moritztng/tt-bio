"""release_train.py: notes, version and the release commit, on strings and a throwaway tree."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("release_train", REPO / "scripts" / "release_train.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

RELEASED = """# Changelog

## [Unreleased]

## [0.14.0] - 2026-10-09

### Changed
- **Old lever.** Faster.
"""
CANDIDATE = """# Changelog

## [Unreleased]

### Changed
- **New lever.** Even faster,
  on two lines.
- **Old lever.** Faster.

### Fixed
- **Old lever.** Faster.

## [0.14.0] - 2026-10-09

### Changed
- **Old lever.** Faster.
"""


def test_notes_drop_what_an_earlier_release_carried_and_empty_headings():
    body = rt.notes(CANDIDATE, RELEASED)
    assert body == "### Changed\n- **New lever.** Even faster,\n  on two lines."


def test_version_is_minor_for_changes_patch_for_fixes_and_skips_taken_tags():
    assert rt.next_version("v0.14.0", "### Changed\n- x", set()) == "0.15.0"
    assert rt.next_version("v0.14.0", "### Fixed\n- x", set()) == "0.14.1"
    assert rt.next_version("v0.14.0", "### Added\n- x", {"v0.15.0"}) == "0.15.1"


def test_release_files_touch_only_the_notes_version_and_install_tag(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(CANDIDATE)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "tt-bio"\nversion = "0.14.0"\n')
    (tmp_path / "README.md").write_text('pip install "x @ git+https://github.com/moritztng/tt-bio.git@v0.14.0"\n')
    rt.release_files(tmp_path, "0.15.0", "### Changed\n- **New lever.**", "2026-10-12")
    cl = (tmp_path / "CHANGELOG.md").read_text()
    assert cl.startswith("# Changelog\n\n## [Unreleased]\n\n## [0.15.0] - 2026-10-12\n\n### Changed\n- **New lever.**\n\n## [0.14.0]")
    assert 'version = "0.15.0"' in (tmp_path / "pyproject.toml").read_text()
    assert "tt-bio.git@v0.15.0" in (tmp_path / "README.md").read_text()
