"""``python -m tt_bio.platform`` — start the ai& Bio web server."""

import argparse

from .app import serve


def main() -> None:
    p = argparse.ArgumentParser(prog="tt_bio.platform", description="Serve the ai& Bio platform.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--workspace", default=None, help="Where to store job working dirs.")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    serve(host=args.host, port=args.port, workspace=args.workspace, debug=args.debug)


if __name__ == "__main__":
    main()
