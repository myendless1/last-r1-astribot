#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import fields
import json

from verl.astribot.deploy.gateway import AstribotGateway, GatewayConfig, serve_gateway


def main() -> None:
    parser = argparse.ArgumentParser(description="Astribot SDK/Quest gateway for LaST-R1 real-robot RL")
    parser.add_argument("--config", help="JSON object or JSON file containing GatewayConfig fields")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--sdk-root")
    parser.add_argument("--quest-state-url")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--enable-robot-commands", action="store_true")
    parser.add_argument("--observation-only", action="store_true")
    args = parser.parse_args()
    payload = {}
    if args.config:
        try:
            payload = json.loads(args.config)
        except json.JSONDecodeError:
            with open(args.config, encoding="utf-8") as stream:
                payload = json.load(stream)
    overrides = {
        "host": args.host, "port": args.port, "sdk_root": args.sdk_root,
        "quest_state_url": args.quest_state_url,
    }
    payload.update({key: value for key, value in overrides.items() if value is not None})
    if args.fake: payload["fake"] = True
    if args.enable_robot_commands: payload["robot_command_enabled"] = True
    if args.observation_only: payload["observation_only"] = True
    allowed = {item.name for item in fields(GatewayConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown: raise ValueError(f"Unknown gateway config fields: {unknown}")
    config = GatewayConfig(**payload)
    print(f"Astribot gateway listening on ws://{config.host}:{config.port}; "
          f"commands={'enabled' if config.robot_command_enabled and not config.observation_only else 'disabled'}")
    serve_gateway(AstribotGateway(config))


if __name__ == "__main__": main()
