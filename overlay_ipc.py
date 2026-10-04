"""
overlay_ipc.py — pushes Nyx's state (idle/listening/thinking/speaking)
plus a live audio level to the NyxOrb quickshell widget over a Unix
socket. Fire-and-forget: if quickshell isn't running or nothing is
connected yet, send_state() is a silent no-op rather than an error.
"""

import json
import os
import socket
import threading

SOCK_PATH = "/tmp/nyx-overlay.sock"

_clients: list[socket.socket] = []
_lock = threading.Lock()
_srv: socket.socket | None = None


def start() -> None:
    """Call once at startup. Opens the socket and accepts connections
    in the background — safe even if NyxOrb isn't running yet, and
    keeps accepting new connections if it reconnects later (e.g. after
    a `qs -c ii` restart)."""
    global _srv
    if os.path.exists(SOCK_PATH):
        os.remove(SOCK_PATH)
    _srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    _srv.bind(SOCK_PATH)
    _srv.listen(5)
    threading.Thread(target=_accept_loop, daemon=True).start()


def _accept_loop() -> None:
    while True:
        try:
            conn, _ = _srv.accept()
        except OSError:
            return
        with _lock:
            _clients.append(conn)


def send_state(state: str, level: float = 0.0) -> None:
    """Push a state update to every connected overlay widget. Drops
    clients that have disconnected; never raises, so a missing/dead
    overlay never interrupts the actual voice pipeline."""
    msg = (json.dumps({"state": state, "level": round(float(level), 3)}) + "\n").encode()
    with _lock:
        dead = []
        for conn in _clients:
            try:
                conn.sendall(msg)
            except OSError:
                dead.append(conn)
        for conn in dead:
            _clients.remove(conn)
