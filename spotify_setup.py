#!/usr/bin/env python3
"""
One-time Spotify credential setup for Nyx. Run this manually once:

    python3 spotify_setup.py

Before running, create a free app at:
    https://developer.spotify.com/dashboard
(no Redirect URI needed — Nyx never asks you to log in; it only uses
the app's own credentials to look up tracks by name).

This just saves your Client ID/Secret to
~/.config/nyx-assistant/spotify.json for main.py to read. Actual
playback happens on your already-running Spotify desktop app via
D-Bus, not through Spotify's servers.
"""

import json
import sys
from pathlib import Path

CONFIG_FILE = Path.home() / ".config" / "nyx-assistant" / "spotify.json"


def main() -> None:
    client_id = input("Spotify Client ID: ").strip()
    client_secret = input("Spotify Client Secret: ").strip()
    if not client_id or not client_secret:
        print("Both fields are required.", file=sys.stderr)
        sys.exit(1)

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps({
        "client_id": client_id,
        "client_secret": client_secret,
    }))
    CONFIG_FILE.chmod(0o600)

    print(f"\nSaved to {CONFIG_FILE}. Try: \"open spotify and play <song>\".")


if __name__ == "__main__":
    main()
