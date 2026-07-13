#!/usr/bin/env python3
from __future__ import annotations

import argparse

from verl.astribot.deploy.robot_io import create_robot, execute_chunk, move_to_initial_pose, read_images, read_state31
from verl.astribot.deploy.transport import pack, unpack


def main():
    from websockets.sync.client import connect
    parser = argparse.ArgumentParser(); parser.add_argument("--uri", default="ws://127.0.0.1:8006")
    parser.add_argument("--prompt", required=True); parser.add_argument("--max-chunks", type=int, default=100)
    parser.add_argument("--execute-count", type=int, default=1); parser.add_argument("--duration", type=float, default=.3)
    parser.add_argument("--init-hdf5"); parser.add_argument("--init-frame-index", type=int, default=0)
    parser.add_argument("--initial-duration", type=float, default=5.0)
    parser.add_argument("--no-warmup", action="store_true"); parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    robot = create_robot(); robot.activate_camera({"left_D405": {"flag_getdepth": False}, "right_D405": {"flag_getdepth": False}, "Bolt": {"flag_getdepth": False}})
    if args.init_hdf5:
        move_to_initial_pose(robot, args.init_hdf5, frame_index=args.init_frame_index, duration=args.initial_duration)
    with connect(args.uri, compression=None, max_size=None) as socket:
        metadata = unpack(socket.recv())["metadata"]; scope = metadata["dimension_scope"]
        def infer():
            request = {"images": read_images(robot), "state": read_state31(robot), "prompt": args.prompt, "image_color_order": "bgr"}
            socket.send(pack(request)); response = unpack(socket.recv())
            if "error" in response: raise RuntimeError(response["error"])
            return response["actions"]
        if not args.no_warmup:
            infer()
        if not args.dry_run and not args.non_interactive:
            input("Policy warmup complete. Press Enter to begin motion (Ctrl-C to abort): ")
        for _ in range(args.max_chunks):
            actions = infer()
            if args.dry_run: print(actions); continue
            execute_chunk(robot, actions, scope=scope, execute_count=args.execute_count, duration=args.duration)


if __name__ == "__main__": main()
