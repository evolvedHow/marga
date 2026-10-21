"""Command-line interface for marga.

Provides the `marga` command with subcommands for running the proxy server.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="marga",
        description="OpenAI-compatible LLM routing proxy",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the proxy server")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Port to listen on (default: 8010)",
    )
    serve_parser.add_argument(
        "--config",
        help="Path to marga.toml configuration file",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (dev mode)",
    )

    args = parser.parse_args()

    if args.command == "serve":
        serve(
            host=args.host,
            port=args.port,
            config=args.config,
            reload=args.reload,
        )
    else:
        parser.print_help()
        sys.exit(1)


def serve(
    host: str = "127.0.0.1",
    port: int = 8010,
    config: str | None = None,
    reload: bool = False,
) -> None:
    """Start the marga proxy server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        config: Path to marga.toml configuration file.
        reload: Enable auto-reload on code changes.
    """
    import os

    import uvicorn

    # Set config path if provided
    if config:
        os.environ["MARGA_CONFIG"] = config

    # Build app from config
    from .proxy import Proxy

    try:
        proxy = Proxy.from_env()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "\nUsage:\n"
            "  1. Create marga.toml from marga.toml.example\n"
            "  2. Set VAULT_URL and VAULT_TOKEN in ~/docker/stack/.env\n"
            "  3. Run: marga serve --config marga.toml\n"
            "\nOr set MARGA_BACKENDS as JSON:\n"
            '  export MARGA_BACKENDS=\'{"openai": {"base_url": "...", ...}}\'\n'
            "  marga serve",
            file=sys.stderr,
        )
        sys.exit(1)

    app = proxy.build_app()

    print(f"Starting marga v{proxy.__class__.__module__.split('.')[0]}")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Backends: {', '.join(proxy.router.backends.keys())}")
    print(f"  Docs: http://{host}:{port}/docs")
    print("\nEndpoints:")
    print(f"  POST http://{host}:{port}/v1/chat/completions  (OpenAI)")
    print(f"  POST http://{host}:{port}/v1/messages          (Anthropic)")
    print(f"  GET  http://{host}:{port}/v1/models")
    print(f"  GET  http://{host}:{port}/health")
    print(f"  GET  http://{host}:{port}/stats")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        reload=reload,
    )


if __name__ == "__main__":
    main()
