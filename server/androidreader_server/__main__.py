"""Entry point: ``python -m androidreader_server``."""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from aiohttp import web

from . import __version__
from .app import make_app, server_urls, viewer_url
from .config import default_data_dir, load_config, rotate_token
from .engines.registry import ENGINES
from .voices import VoiceStore

VOICESTUDIO_DB = Path(os.path.expandvars(r"%APPDATA%\OmniVoice\omnivoice.db"))
VOICESTUDIO_VOICES = Path(os.path.expandvars(r"%APPDATA%\OmniVoice\voices"))


def _print_addresses(cfg) -> None:
    urls = server_urls(cfg.port, cfg.extra_urls)
    print(f"\nAndroidReader server {__version__}")
    print(f"  Listening on   : {cfg.host}:{cfg.port}")
    print(f"  Models         : {cfg.models_path}")
    print(f"  Default engine : {cfg.default_engine}")
    print(f"  Token          : {cfg.token}")
    if not urls:
        print("  No network address found - connect this PC to a network.")
        return
    print(f"  Library        : http://127.0.0.1:{cfg.port}/books")
    for url in urls:
        print(f"  On the network : {url}/books")
    print(f"\nLinks and QR codes for other devices: open http://127.0.0.1:{cfg.port}/ on this PC.")
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(viewer_url(urls[0], cfg.token))
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(f"Or scan this to open the library ({urls[0]}):")
        qr.print_ascii(out=sys.stdout, invert=True)
    except Exception:  # noqa: BLE001 - the QR is a convenience only
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="androidreader_server", description=__doc__)
    parser.add_argument("--host", help="bind address (default from config: 0.0.0.0)")
    parser.add_argument("--port", type=int, help="port (default from config: 3950)")
    parser.add_argument("--engine", help=f"default engine ({', '.join(ENGINES)})")
    parser.add_argument("--no-preload", action="store_true", help="don't load the default engine at start-up")
    parser.add_argument("--rotate-token", action="store_true", help="generate a new token and exit")
    parser.add_argument("--show-links", action="store_true", help="print the library links and exit")
    parser.add_argument(
        "--import-voicestudio", action="store_true",
        help="copy VoiceStudio voice profiles (read-only) into the voice store and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    data_dir = default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    file_log = logging.handlers.RotatingFileHandler(
        data_dir / "server.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
    )
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_log)
    cfg = rotate_token(data_dir) if args.rotate_token else load_config(data_dir)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.engine:
        cfg.default_engine = args.engine
    if args.no_preload:
        cfg.preload = False

    if args.import_voicestudio:
        store = VoiceStore(data_dir / "voices")
        if not VOICESTUDIO_DB.exists():
            print(f"VoiceStudio database not found at {VOICESTUDIO_DB}")
            return 1
        imported = store.import_omnivoice(VOICESTUDIO_DB, VOICESTUDIO_VOICES)
        for v in imported:
            print(f"  imported {v.id}  {v.name}")
        print(f"{len(imported)} voice(s) imported; {len(store.list())} in the store.")
        return 0

    _print_addresses(cfg)
    if args.rotate_token or args.show_links:
        return 0
    print("\nPress Ctrl+C to stop.\n")
    web.run_app(
        make_app(cfg),
        host=cfg.host,
        port=cfg.port,
        handler_cancellation=True,
        access_log=None,
        print=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
