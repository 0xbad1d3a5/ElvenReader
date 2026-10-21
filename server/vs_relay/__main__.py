"""Entry point: ``python -m vs_relay``."""
from __future__ import annotations

import argparse
import logging
import sys

from aiohttp import web

from . import __version__
from .app import make_app
from .config import default_data_dir, load_config, rotate_token
from .netinfo import lan_addresses


def _print_pairing(cfg) -> None:
    addresses = lan_addresses()
    print(f"\nVoiceStudio TTS relay {__version__}")
    print(f"  VoiceStudio backend : {cfg.voicestudio_url}")
    print(f"  Listening on        : {cfg.host}:{cfg.port}")
    print(f"  Token               : {cfg.token}")
    if not addresses:
        print("  No LAN address found - connect this PC to a network.")
        return
    for addr in addresses:
        print(f"  Phone server URL    : http://{addr['ip']}:{cfg.port}   ({addr['kind']})")
    print(f"\nPairing QR codes: open http://127.0.0.1:{cfg.port}/ in a browser on this PC.")
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(f"http://{addresses[0]['ip']}:{cfg.port}/pair#{cfg.token}")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(f"Or scan this ({addresses[0]['ip']}):")
        qr.print_ascii(out=sys.stdout, invert=True)
    except Exception:  # noqa: BLE001 - the QR is a convenience only
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vs_relay", description=__doc__)
    parser.add_argument("--host", help="bind address (default from config: 0.0.0.0)")
    parser.add_argument("--port", type=int, help="port (default from config: 3950)")
    parser.add_argument("--voicestudio-url", help="VoiceStudio backend URL (default http://127.0.0.1:3900)")
    parser.add_argument("--rotate-token", action="store_true", help="generate a new token and exit")
    parser.add_argument("--show-pairing", action="store_true", help="print pairing info and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    data_dir = default_data_dir()
    cfg = rotate_token(data_dir) if args.rotate_token else load_config(data_dir)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.voicestudio_url:
        cfg.voicestudio_url = args.voicestudio_url.rstrip("/")

    _print_pairing(cfg)
    if args.rotate_token or args.show_pairing:
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
