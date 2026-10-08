"""A module-level import that is repeated inside a function becomes a local name for that whole
function, so every use of it ABOVE the inner import raises UnboundLocalError. That is what broke
AFTER's host arm after the sidecar landed: `import json` inside main() turned the module's own
`import json` into a local, and the `json.dumps` 38 lines earlier died on the first call.

Nothing in the normal test path imports these scripts -- they are argparse entry points that build
a predictor -- so the failure only showed up hours into a campaign launch. This reads them as
source instead.
"""

import ast
import pathlib

import pytest

SCRIPTS = sorted((pathlib.Path(__file__).resolve().parents[1] / "perf" / "bci_accept").glob("*.py"))


def shadowed_imports(source, filename="<string>"):
    """Names imported at module level and imported again inside a function in the same file."""
    tree = ast.parse(source, filename=filename)

    module_level = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_level.add(alias.asname or alias.name.split(".")[0])

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if inner is node or not isinstance(inner, (ast.Import, ast.ImportFrom)):
                continue
            for alias in inner.names:
                name = alias.asname or alias.name.split(".")[0]
                if name in module_level:
                    found.append((node.name, name, inner.lineno))
    return found


def test_the_checker_catches_the_bug_it_was_written_for():
    source = "import json\n\n\ndef main():\n    print(json.dumps({}))\n    import json\n"
    assert shadowed_imports(source) == [("main", "json", 6)]


def test_the_checker_allows_an_import_that_is_only_local():
    source = "import json\n\n\ndef main():\n    import csv\n    return csv, json\n"
    assert shadowed_imports(source) == []


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_script_shadows_its_own_module_level_import(script):
    found = shadowed_imports(script.read_text(), filename=str(script))
    assert not found, "\n".join(
        f"{script.name}:{lineno}: {func}() re-imports {name}, which is already imported at module "
        f"level -- every use of {name} earlier in {func}() will raise UnboundLocalError"
        for func, name, lineno in found
    )
