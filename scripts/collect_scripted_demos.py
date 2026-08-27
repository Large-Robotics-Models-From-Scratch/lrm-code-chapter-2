#!/usr/bin/env python
"""Record scripted pick-and-place demonstrations as a LeRobot dataset.

The scripted policy in `ch02.scripted` solves roughly half of its
episodes, which is enough to be a demonstrator: we keep only the
successful ones and write them in the same format as the SO-101 teleop
datasets, so a LeRobot policy can be fine-tuned on them directly.

Training a policy on this and evaluating it in the same simulator tests
the pipeline and whether the architecture can learn the task. It says
nothing about sim-to-real -- both sides are simulation.

    python scripts/collect_scripted_demos.py --episodes 100
"""

import argparse

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ch02.scripted import (
    IKMotionPlanner,
    _compute_top_down_grasp_pose,
    run_scripted_episode,
)

CAMERAS = ("up", "side")
IMG_HW = 256          # what SmolVLA resizes to; keeps the set small
TASK = "Grasp the cube and put it in the target position"


def features(img_hw: int) -> dict:
    feats = {
        "observation.state": {
            "dtype": "float32", "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                      "wrist_flex", "wrist_roll", "gripper"],
        },
        "action": {
            "dtype": "float32", "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                      "wrist_flex", "wrist_roll", "gripper"],
        },
    }
    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (img_hw, img_hw, 3),
            "names": ["height", "width", "channel"],
        }
    return feats


def _small_rgb(sensor_data, cam: str, img_hw: int) -> np.ndarray:
    """One camera's frame as HWC uint8, resized for training."""
    t = sensor_data[cam]["rgb"]
    t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    t = t.permute(0, 3, 1, 2).float()
    t = torch.nn.functional.interpolate(
        t, size=(img_hw, img_hw), mode="bilinear", align_corners=False
    )
    arr = t[0].permute(1, 2, 0).cpu().numpy()
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


class Tape:
    """Capture (observation, action) at every env.step the planner makes."""

    def __init__(self, env, img_hw: int):
        self.env, self.img_hw = env, img_hw
        self.u = env.unwrapped
        self.frames: list[dict] = []
        self._orig = self.u.step

    def __enter__(self):
        def step(action):
            obs = self.u.get_obs()
            row = {
                "observation.state": np.asarray(
                    obs["agent"]["qpos"].cpu()
                ).reshape(-1)[:6].astype(np.float32),
                "action": np.asarray(action).reshape(-1)[:6].astype(
                    np.float32
                ),
            }
            for cam in CAMERAS:
                row[f"observation.images.{cam}"] = _small_rgb(
                    obs["sensor_data"], cam, self.img_hw
                )
            self.frames.append(row)
            return self._orig(action)

        self.u.step = step
        return self

    def __exit__(self, *exc):
        self.u.step = self._orig


def collect(n_episodes: int, repo_id: str, img_hw: int, seed0: int) -> None:
    env = gym.make(
        "PickCubeSO101-v1", obs_mode="rgb", control_mode="pd_joint_pos",
        max_episode_steps=400, render_mode=None,
    )
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=20, features=features(img_hw),
        robot_type="so101", use_videos=True,
    )

    kept = attempted = 0
    seed = seed0
    while kept < n_episodes:
        attempted += 1
        env.reset(seed=seed)
        planner = IKMotionPlanner(env)
        with Tape(env, img_hw) as tape:
            try:
                grasp = _compute_top_down_grasp_pose(env)
                goal = env.unwrapped.goal_site.pose.sp.p
                run_scripted_episode(planner, grasp, np.asarray(goal))
            except Exception as exc:  # noqa: BLE001
                print(f"  seed {seed}: planner error "
                      f"{type(exc).__name__}: {exc}")
                seed += 1
                continue
        info = env.unwrapped.evaluate()
        ok = bool(np.asarray(
            info.get("success", False).cpu()
            if hasattr(info.get("success", False), "cpu")
            else info.get("success", False)
        ).reshape(-1)[0])
        if ok:
            for row in tape.frames:
                ds.add_frame({**row, "task": TASK})
            ds.save_episode()
            kept += 1
            print(f"  seed {seed}: kept ({kept}/{n_episodes}, "
                  f"{len(tape.frames)} frames)")
        else:
            print(f"  seed {seed}: discarded (unsuccessful)")
        seed += 1

    env.close()
    print(f"\n{kept} episodes kept from {attempted} attempts "
          f"({kept / attempted:.0%}) -> {ds.root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--repo-id", default="local/so101_scripted_pickplace")
    ap.add_argument("--img-hw", type=int, default=IMG_HW)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    collect(args.episodes, args.repo_id, args.img_hw, args.seed0)
