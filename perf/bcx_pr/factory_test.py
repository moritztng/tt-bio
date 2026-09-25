#!/usr/bin/env python3
"""Check the predictor-factory change: identical behaviour when the setting is absent, and both
construction sites resolved once.

Runs without jax, because importing `bindcraft.campaign` pulls the whole JAX stack and the thing
under test is 15 lines of plain Python. The resolver's source is lifted out of campaign.py by AST
and executed against a sentinel class, so it is the code as written in the file rather than a copy.

Run:  python3 perf/bcx_pr/factory_test.py <path-to-patched-BindCraft2-checkout>
"""
import ast
import dis
import sys
from pathlib import Path

WANTED = {"DESIGN_MODEL_FACTORIES", "register_design_model", "design_model_factory"}


def resolver_namespace(campaign: Path) -> dict:
    tree = ast.parse(campaign.read_text())
    picked = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in WANTED)
        or (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) in WANTED)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in WANTED for t in node.targets))
    ]
    names = {getattr(node, "name", None) or ast.unparse(node).split(":")[0].split("=")[0].strip() for node in picked}
    missing = WANTED - names
    assert not missing, f"campaign.py is missing {sorted(missing)}"
    namespace = {"AlphaFoldDesignModel": Sentinel}
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(campaign), "exec"), namespace)
    return namespace


class Sentinel:
    pass


class OtherBackend:
    pass


def main(root: Path) -> int:
    campaign = root / "bindcraft" / "campaign.py"
    ns = resolver_namespace(campaign)
    factory, register = ns["design_model_factory"], ns["register_design_model"]

    checks = []
    checks.append(("absent setting -> AlphaFoldDesignModel", factory({}) is Sentinel))
    checks.append(("unrelated settings -> AlphaFoldDesignModel", factory({"design_recycles": 1}) is Sentinel))
    checks.append(("design_model=None -> AlphaFoldDesignModel", factory({"design_model": None}) is Sentinel))
    checks.append(("design_model='alphafold2' -> AlphaFoldDesignModel", factory({"design_model": "alphafold2"}) is Sentinel))

    register("other", OtherBackend)
    checks.append(("registered name -> that factory", factory({"design_model": "other"}) is OtherBackend))
    try:
        factory({"design_model": "nope"})
        raised = False
    except ValueError as err:
        raised = "nope" in str(err) and "other" in str(err)
    checks.append(("unknown name -> ValueError naming it and the registry", raised))

    # both construction sites must come from ONE resolution, or design and validation can differ
    source = campaign.read_text()
    run_campaign = next(
        code
        for code in compile(source, str(campaign), "exec").co_consts
        if hasattr(code, "co_code") and code.co_name == "run_campaign"
    )
    lambda_code = next(code for code in run_campaign.co_consts if hasattr(code, "co_code") and code.co_name == "<lambda>")
    direct = [
        ins.opname
        for ins in dis.get_instructions(run_campaign)
        if ins.argval == "build_design_model" and ins.opname.startswith("LOAD_")
    ]
    closed = [ins.opname for ins in dis.get_instructions(lambda_code) if ins.argval == "build_design_model"]
    leftover = [
        ins.positions.lineno
        for code in (run_campaign, lambda_code)
        for ins in dis.get_instructions(code)
        if ins.argval == "AlphaFoldDesignModel"
    ]
    checks.append(("design site uses the resolved callable", bool(direct)))
    checks.append(("validation lambda closes over the SAME callable", any(op.startswith("LOAD_DEREF") for op in closed)))
    checks.append(("no construction bypasses the resolver", not leftover))

    for label, ok in checks:
        print(f"  {'pass' if ok else 'FAIL'}  {label}")
    failed = [label for label, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".bc2")))
