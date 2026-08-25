"""Minimal example subscriber for the HandStatePublisher UDP stream.

Uses only Python's standard library (socket, json, argparse) -- copy this single
file into any environment (no dex_retargeting / mujoco / numpy required) to consume
the wrist pose + finger joint angle stream published by my_retargeting_mujoco.py
when `simulation.socket_publish.enabled: true`.

Usage:
    python state_subscriber_example.py --port 6001
    # Same-machine loopback also works fine since 0.0.0.0 covers 127.0.0.1 too.

To receive from another machine on the LAN, see MyREADME.md section 8: point the
publisher's `simulation.socket_publish.host` at this machine's LAN IP (or a
broadcast address), and make sure the receiving machine's firewall allows inbound
UDP on this port. `--host` defaults to 0.0.0.0 (all interfaces) precisely so this
script works unmodified for that case.
"""

from __future__ import annotations

import argparse
import json
import socket
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind. 0.0.0.0 (default) accepts packets from any "
        "interface, including localhost -- use this for both same-machine and "
        "cross-LAN listening. Only narrow this if you specifically need to "
        "restrict which interface receives packets.",
    )
    parser.add_argument("--port", type=int, default=6001)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"Listening for hand-state UDP packets on {args.host}:{args.port} ...")

    last_print = 0.0
    while True:
        raw, _addr = sock.recvfrom(65536)
        msg = json.loads(raw.decode("utf-8"))

        now = time.time()
        if now - last_print < 0.2:  # throttle printing to ~5 Hz for readability
            continue
        last_print = now

        wrist_pos = msg["wrist_pos"]
        wrist_quat = msg["wrist_quat_wxyz"]
        print(
            f"[{msg['hand']}] sim_t={msg['sim_time']:.3f} "
            f"wrist_pos=[{', '.join(f'{v:.3f}' for v in wrist_pos)}] "
            f"wrist_quat_wxyz=[{', '.join(f'{v:.3f}' for v in wrist_quat)}]"
        )
        for name, angle in zip(msg["joint_names"], msg["joint_angles"]):
            print(f"    {name}: {angle:.4f} rad")


if __name__ == "__main__":
    main()
