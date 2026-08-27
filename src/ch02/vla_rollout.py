"""Run a LeRobot vision-language-action policy in the SO-101 simulator.

Bridges the two conventions that differ between `PickCubeSO101-v1` and the
`lerobot/svla_so101_pickplace` teleop datasets a SmolVLA checkpoint is
trained on:

- **state and action units** — the sim speaks radians, LeRobot speaks
  normalized motor units, so both conversions run through the
  dataset's own statistics.
- **gripper sense** — SO-101's sim joint runs `-0.1745` (closed) to
  `+1.7453` (open). If a checkpoint was trained with the opposite
  convention, flip it with `invert_gripper=True`.

The camera names already line up: the env publishes `up` and `side` at
640x480 to match the dataset, so images map across untouched.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch

SIM_GRIPPER_CLOSED = -0.1745
SIM_GRIPPER_OPEN = 1.7453
CAMERAS = ("up", "side")
DEFAULT_TASK = "Grasp the cube and put it in the target position"


def _to_chw_float(img: Any) -> torch.Tensor:
    """(1,H,W,3) uint8 sim image -> (1,3,H,W) float32 in [0, 1]."""
    t = img if isinstance(img, torch.Tensor) else torch.as_tensor(img)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    t = t.permute(0, 3, 1, 2).float()
    if t.max() > 1.5:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


class UnitBridge:
    """Convert joint vectors between sim radians and LeRobot units.

    LeRobot policies are trained on normalized data, so the mapping runs
    through the dataset's per-dimension statistics rather than a constant
    scale factor.
    """

    def __init__(self, stats=None, invert_gripper: bool = False):
        self.stats = stats
        self.invert_gripper = invert_gripper

    @staticmethod
    def _get(stats_entry, key):
        v = stats_entry[key]
        return torch.as_tensor(np.asarray(v), dtype=torch.float32)

    def sim_to_policy(self, qpos: torch.Tensor) -> torch.Tensor:
        """Sim joint positions -> the policy's state vector."""
        q = torch.as_tensor(qpos, dtype=torch.float32).reshape(1, -1)[:, :6]
        if self.invert_gripper:
            q = q.clone()
            q[:, 5] = SIM_GRIPPER_OPEN + SIM_GRIPPER_CLOSED - q[:, 5]
        if self.stats is None:
            return q
        s = self.stats["observation.state"]
        return (q - self._get(s, "mean")) / (self._get(s, "std") + 1e-8)

    def policy_to_sim(self, action: torch.Tensor) -> np.ndarray:
        """Policy action -> a `pd_joint_pos` sim action."""
        a = torch.as_tensor(action, dtype=torch.float32)
        a = a.reshape(1, -1)[:, :6]
        if self.stats is not None:
            s = self.stats["action"]
            a = a * (self._get(s, "std") + 1e-8) + self._get(s, "mean")
        a = a.squeeze(0).cpu().numpy().astype(np.float32)
        if self.invert_gripper:
            a[5] = SIM_GRIPPER_OPEN + SIM_GRIPPER_CLOSED - a[5]
        return a


def load_policy(repo_id: str, device: str = "cuda"):
    """Load a LeRobot policy together with its pre/post processors.

    The processors are not optional decoration: they tokenize the task
    string into `observation.language.tokens` and apply the checkpoint's
    own normalization statistics. Calling `select_action` on a raw batch
    fails without them.
    """
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(repo_id).eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=repo_id
    )
    return policy, preprocessor, postprocessor


def policy_image_spec(policy) -> tuple[list, tuple]:
    """What image keys and resolution a checkpoint actually expects.

    Checkpoints differ: `smolvla_base` wants `camera1/2/3` at 256x256,
    while one fine-tuned on `svla_so101_pickplace` wants `up` and `side`.
    Read it off the config instead of assuming our camera names match.
    """
    feats = getattr(policy.config, "input_features", {})
    keys, shape = [], (3, 256, 256)
    for name, feat in feats.items():
        if "image" in name:
            keys.append(name)
            shape = tuple(feat.shape)
    return keys, shape


def build_batch(
    obs: dict,
    bridge: UnitBridge,
    task: str,
    image_keys=None,
    image_size=None,
) -> dict:
    """Assemble one LeRobot-style batch from a sim observation.

    Our two sim views are mapped onto whatever image slots the policy
    declares, cycling if it wants more than we have.
    """
    views = [
        _to_chw_float(obs["sensor_data"][cam]["rgb"]) for cam in CAMERAS
    ]
    if image_size is not None:
        h, w = image_size[-2], image_size[-1]
        views = [
            torch.nn.functional.interpolate(
                v, size=(h, w), mode="bilinear", align_corners=False
            )
            for v in views
        ]
    keys = image_keys or [f"observation.images.{c}" for c in CAMERAS]
    batch = {k: views[i % len(views)] for i, k in enumerate(keys)}
    batch["observation.state"] = bridge.sim_to_policy(obs["agent"]["qpos"])
    batch["task"] = [task]
    return batch


def run_vla_episode(
    policy,
    env: gym.Env,
    bridge: UnitBridge,
    task: str = DEFAULT_TASK,
    max_steps: int = 300,
    seed: int = 0,
    preprocessor=None,
    postprocessor=None,
) -> dict:
    """Roll one episode of `policy` in `env`.

    Returns a summary dict with the outcome and how close the cube got.
    """
    obs, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    device = next(policy.parameters()).device
    image_keys, image_size = policy_image_spec(policy)
    if hasattr(policy, "reset"):
        policy.reset()

    best = float("inf")
    info: dict = {}
    for _ in range(max_steps):
        batch = build_batch(
            obs, bridge, task, image_keys, image_size
        )
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        if preprocessor is not None:
            batch = preprocessor(batch)
        with torch.no_grad():
            action = policy.select_action(batch)
        if postprocessor is not None:
            action = postprocessor(action)
        sim_action = bridge.policy_to_sim(action)
        obs, _, terminated, truncated, info = env.step(sim_action)
        cube = np.asarray(unwrapped.cube.pose.sp.p)
        goal = np.asarray(unwrapped.goal_site.pose.sp.p)
        best = min(best, float(np.linalg.norm(cube - goal)))
        if bool(np.asarray(terminated).reshape(-1)[0]):
            break

    def flag(key):
        v = info.get(key, False)
        v = v.cpu() if hasattr(v, "cpu") else v
        return bool(np.asarray(v).reshape(-1)[0])

    return dict(
        seed=seed,
        success=flag("success"),
        is_grasped=flag("is_grasped"),
        closest_cube_to_goal_mm=round(best * 1000, 1),
    )
