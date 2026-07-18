from __future__ import annotations

from typing import Any

from .transport import pack, unpack


class AstribotGatewayClient:
    def __init__(self, uri: str, *, timeout: float = 30.0):
        from websockets.sync.client import connect
        self.socket = connect(uri, open_timeout=timeout, close_timeout=timeout, max_size=None, compression=None)
        self.metadata = self.call("hello")["metadata"]

    def call(self, operation: str, **payload) -> dict[str, Any]:
        self.socket.send(pack({"op": operation, **payload}))
        response = unpack(self.socket.recv())
        if response.get("error") and operation not in {"step", "observe", "hold"}:
            raise RuntimeError(response["error"])
        return response

    def reset(self, *, confirm: bool = True): return self.call("reset", confirm=confirm)
    def observe(self): return self.call("observe")
    def step(self, actions): return self.call("step", actions=actions)
    def hold(self): return self.call("hold")
    def close(self):
        try:
            self.call("close")
        except Exception:
            pass
        finally:
            self.socket.close()

    def assert_compatible(self, protocol) -> None:
        expected = {"protocol_version": 1, "dimension_scope": protocol.dimension_scope,
                    "state_dim": protocol.state_dim, "action_dim": protocol.action_dim,
                    "image_color_order": "bgr"}
        mismatches = [f"{key}: expected {value!r}, got {self.metadata.get(key)!r}"
                      for key, value in expected.items() if self.metadata.get(key) != value]
        if mismatches: raise ValueError("Incompatible Astribot gateway: " + "; ".join(mismatches))
