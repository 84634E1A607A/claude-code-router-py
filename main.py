"""Entry point — load config from --config, CCR_CONFIG, or config.json."""

import argparse
import logging
import os
import sys

import uvicorn

from config import load_config
from server import app, set_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code Router (Python)")
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="Load config from a JSON file")
    args = parser.parse_args()

    config_path = args.config or os.environ.get("CCR_CONFIG", "config.json")

    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        sys.exit(1)

    host = "0.0.0.0"
    port = int(cfg.get("PORT", 3456))
    log_level = "info"

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Starting Claude Code Router on {host}:{port}", flush=True)

    set_config(cfg)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
