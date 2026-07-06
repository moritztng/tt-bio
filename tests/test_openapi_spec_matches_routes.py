"""Guard the hand-authored OpenAPI spec against drifting from the real /v1 routes.

openapi_spec.build_spec() is written by hand (deliberately — the surface is small
and stable). The risk is that a route gets added/removed/renamed without the spec
following. This asserts the two agree on the set of /v1 paths, comparing them
structurally (path parameters normalized) so internal Flask param names don't
have to match the spec's public template names.
"""
from __future__ import annotations

import re

from flask import Flask

from tt_bio.platform.api_v1 import bp
from tt_bio.platform.openapi_spec import build_spec

# The spec-serving endpoint documents the others, not itself.
_NOT_IN_SPEC = {"/v1/openapi.json"}


def _norm(path: str) -> str:
    """Collapse any <flask:param> or {openapi_param} to a placeholder."""
    return re.sub(r"[<{][^>}]+[>}]", "{}", path)


def test_spec_paths_match_v1_routes():
    app = Flask(__name__)
    app.register_blueprint(bp)
    route_paths = {
        _norm(str(r)) for r in app.url_map.iter_rules()
        if str(r).startswith("/v1") and str(r) not in _NOT_IN_SPEC
    }
    spec_paths = {_norm(p) for p in build_spec("test")["paths"]}
    assert route_paths == spec_paths, (
        f"OpenAPI spec drifted from routes.\n"
        f"  routes not in spec: {sorted(route_paths - spec_paths)}\n"
        f"  spec not in routes: {sorted(spec_paths - route_paths)}"
    )
