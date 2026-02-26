"""Entry point — load config and start the server."""

import argparse
import logging
import sys

import uvicorn

from config import load_config
from server import app, set_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code Router (Python)")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument("--host", default=None, help="Override listen host")
    parser.add_argument("--port", type=int, default=None, help="Override listen port")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["debug", "info", "warning", "error"],
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        sys.exit(1)

    set_config(cfg)

    host = args.host or cfg.get("HOST", "0.0.0.0")
    port = args.port or cfg.get("PORT", 3456)
    log_level = args.log_level or cfg.get("LOG_LEVEL", "info")

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Starting Claude Code Router on {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
