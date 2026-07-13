from __future__ import annotations

import argparse
import asyncio

from .policy import AstribotPolicy
from .transport import pack, unpack


async def serve(policy, host: str, port: int):
    import websockets
    async def handler(socket):
        await socket.send(pack({"metadata": policy.metadata()}))
        async for request in socket:
            try: response = policy.infer(unpack(request))
            except Exception as error: response = {"error": f"{type(error).__name__}: {error}"}
            await socket.send(pack(response))
    async with websockets.serve(handler, host, port, compression=None, max_size=None): await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--norm", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--host", default="0.0.0.0"); parser.add_argument("--port", type=int, default=8006)
    args = parser.parse_args(); asyncio.run(serve(AstribotPolicy(args.checkpoint, args.norm, args.device), args.host, args.port))


if __name__ == "__main__": main()
