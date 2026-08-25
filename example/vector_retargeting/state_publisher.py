"""Fire-and-forget UDP JSON publisher for the control hand's live wrist pose + finger
joint angles.

One JSON object per UDP datagram; a subscriber only needs Python's standard library
(`socket` + `json`), so it can run in a completely different environment/dependency
set than this repo. See `state_subscriber_example.py` for a minimal consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import json
import socket
import time


@dataclass
class HandStatePublisher:
    host: str
    port: int
    _sock: Optional[socket.socket] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Harmless for normal unicast/loopback use; lets `host` be set to a broadcast
        # address (e.g. "192.168.1.255" or "255.255.255.255") to reach every listener
        # on the LAN without hardcoding a specific machine's IP.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def publish(
        self,
        hand: str,
        sim_time: float,
        wrist_pos: Sequence[float],
        wrist_quat_wxyz: Sequence[float],
        joint_names: Sequence[str],
        joint_angles: Sequence[float],
    ) -> None:
        if self._sock is None:
            return
        payload = {
            "t": time.time(),
            "sim_time": float(sim_time),
            "hand": hand,
            "wrist_pos": [float(x) for x in wrist_pos],
            "wrist_quat_wxyz": [float(x) for x in wrist_quat_wxyz],
            "joint_names": list(joint_names),
            "joint_angles": [float(x) for x in joint_angles],
        }
        try:
            self._sock.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))
        except OSError:
            # Best-effort: a publish hiccup (e.g. no listener yet) must never break the sim loop.
            pass

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
