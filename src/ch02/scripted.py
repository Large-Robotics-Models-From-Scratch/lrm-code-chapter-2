"""Multi-phase scripted pick-and-place via SAPIEN inverse kinematics.

The seven phases — approach, descend, grasp, lift, transport, place, hold —
are expressed as Cartesian keyframe poses. An inlined `IKMotionPlanner`
solves IK against SAPIEN's pinocchio model, interpolates a Cartesian
waypoint trajectory between the current and target end-effector poses,
and feeds each waypoint as a `pd_joint_pos` action to the env.

We do not use ManiSkill's mplib-based motion planner: mplib 0.1.1
segfaults under numpy 2.x at `ArticulatedModel` construction. SAPIEN's
pinocchio model ships with the simulator and avoids that dependency
chain entirely.

The policy plateaus around 10–20% success — the SO-100 arm's reach is
marginal for some goal positions, and our IK + linear-interpolation
path can't always thread the workspace as cleanly as a sampling-based
planner. That gap is the chapter's motivation for learned policies.
"""

import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from ch02._metrics import _episode_success

# A few canonical SO-100 joint-space warm starts for IK. CLIK can get
# trapped near singularities; retrying from a different seed often
# escapes them.
_IK_WARM_STARTS = [
    np.array([0.0, 0.0, 0.0, 1.6, 1.6, 0.0]),
    np.array([0.0, 0.5, -0.5, 1.6, 0.0, 0.0]),
    np.array([0.0, 0.5, -0.5, 1.6, np.pi, 0.0]),
    np.array([0.0, -0.5, 0.5, 1.6, 0.0, 0.0]),
]


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    return s0 * q0 + s1 * q1


class IKMotionPlanner:
    """SAPIEN-IK-driven planner with the same API as mplib's solver.

    Exposes `move_to_pose_with_screw`, `close_gripper`, `open_gripper`,
    `hold`, and `close` so the same `run_scripted_episode` body works.
    Requires the env to be constructed with `control_mode="pd_joint_pos"`.
    """

    OPEN = 0.0
    CLOSED = -0.8
    TIP_LINK_NAME = "Fixed_Jaw_tip"
    GRIPPER_JOINT_IDX = 5
    IK_ACCEPT_TOL = 0.10  # 10cm; skip waypoints whose IK diverges further

    def __init__(
        self,
        env: gym.Env,
        n_waypoints: int = 30,
        ik_max_iter: int = 200,
        settle_steps: int = 10,
    ):
        self.env = env
        self.unwrapped = env.unwrapped
        self.agent = self.unwrapped.agent
        self.robot_wrap = self.agent.robot
        self.robot = self.robot_wrap._objs[0]  # SAPIEN PhysxArticulation
        self.pino = self.robot.create_pinocchio_model()

        # Locate Fixed_Jaw_tip's link index for IK. The link name is
        # prefixed by ManiSkill's scene id, so match by suffix.
        link_names = [link.get_name() for link in self.robot.get_links()]
        for i, n in enumerate(link_names):
            if n.endswith(self.TIP_LINK_NAME):
                self.tip_link_idx = i
                break
        else:
            raise RuntimeError(
                f"Could not find {self.TIP_LINK_NAME} link"
            )

        # Per-joint position limits — SAPIEN's IK doesn't enforce these;
        # we clamp the IK result before sending so the controller doesn't
        # silently clip it.
        joints = self.robot.get_active_joints()
        self.q_low = np.array(
            [j.get_limits()[0][0] for j in joints], dtype=np.float64
        )
        self.q_high = np.array(
            [j.get_limits()[0][1] for j in joints], dtype=np.float64
        )

        self.n_waypoints = n_waypoints
        self.ik_max_iter = ik_max_iter
        self.settle_steps = settle_steps
        self.gripper_state = self.OPEN

    def close(self) -> None:
        """No-op; matches the mplib planner API."""
        pass

    def _current_qpos(self) -> np.ndarray:
        return np.array(self.robot.qpos, dtype=np.float64).copy()

    def _clamp_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qp = np.asarray(qpos, dtype=np.float64).copy()
        for i in range(6):
            qp[i] = max(self.q_low[i], min(self.q_high[i], qp[i]))
        return qp

    def _step_with_qpos(self, qpos_target: np.ndarray):
        action = qpos_target[:6].copy().astype(np.float32)
        # Override gripper command with our tracked open/close state so
        # the controller doesn't fight the policy's grip during transport.
        action[self.GRIPPER_JOINT_IDX] = self.gripper_state
        return self.env.step(action)

    def _current_tip_pose_world(self) -> sapien.Pose:
        return self.robot_wrap.links_map[self.TIP_LINK_NAME].pose.sp

    def _tcp_to_tip_world(self, tcp_pose: sapien.Pose) -> sapien.Pose:
        """Convert target TCP pose → equivalent Fixed_Jaw_tip pose.

        TCP is the midpoint of the two jaw tips; Fixed_Jaw_tip is one of
        them. We approximate the (tip → midpoint) displacement using the
        current world-frame offset, which holds to mm-level across our
        sequential gripper-state transitions.
        """
        tip_now = self._current_tip_pose_world()
        tcp_pos = self.agent.tcp_pos
        if hasattr(tcp_pos, "cpu"):
            tcp_pos = tcp_pos.cpu().numpy()
        tcp_now_p = np.asarray(tcp_pos).reshape(-1)[:3]
        offset = np.asarray(tip_now.p) - tcp_now_p
        return sapien.Pose(
            p=(np.asarray(tcp_pose.p) + offset).tolist(),
            q=tcp_pose.q,
        )

    def _solve_ik(self, target_base, warm_start):
        """One CLIK call. Returns `(qpos, err_norm)`."""
        qp, _, err = self.pino.compute_inverse_kinematics(
            self.tip_link_idx,
            target_base,
            initial_qpos=warm_start.astype(np.float64),
            max_iterations=self.ik_max_iter,
        )
        return qp, float(np.linalg.norm(err))

    def move_to_pose_with_screw(self, target_tcp_pose: sapien.Pose) -> bool:
        """Cartesian waypoint interpolation + per-waypoint IK.

        Linearly interpolates `n_waypoints` poses in SE(3) from the
        current Fixed_Jaw_tip pose to the target, solves IK at each
        waypoint with the previous solution as warm start, clamps to
        joint limits, and sends each clamped qpos as a `pd_joint_pos`
        action. Mirrors mplib's screw-motion semantics without the
        sampling-based planner.

        Args:
            target_tcp_pose: Desired TCP pose in world frame.

        Returns:
            Always True; here for parity with planners that may abort.
        """
        start = self._current_tip_pose_world()
        end = self._tcp_to_tip_world(target_tcp_pose)
        p0 = np.asarray(start.p)
        p1 = np.asarray(end.p)
        q0 = np.asarray(start.q)
        q1 = np.asarray(end.q)
        base_inv = self.robot.pose.inv()

        warm = self._current_qpos()
        for i in range(1, self.n_waypoints + 1):
            alpha = i / self.n_waypoints
            wp_p = p0 + alpha * (p1 - p0)
            wp_q = _slerp(q0, q1, alpha)
            wp_world = sapien.Pose(p=wp_p.tolist(), q=wp_q.tolist())
            wp_base = base_inv * wp_world

            qpos_target, err = self._solve_ik(wp_base, warm)
            # Retry from canonical warm starts if chained solve diverged.
            if err > self.IK_ACCEPT_TOL:
                for ws in _IK_WARM_STARTS:
                    cand, cand_err = self._solve_ik(wp_base, ws)
                    if cand_err < err:
                        qpos_target, err = cand, cand_err
                    if err < self.IK_ACCEPT_TOL / 2:
                        break

            if err > self.IK_ACCEPT_TOL:
                continue

            clamped = self._clamp_qpos(qpos_target)
            self._step_with_qpos(clamped)
            warm = clamped.copy()
        return True

    def close_gripper(self, gripper_state: float = -0.8) -> None:
        """Close the gripper; step a few times so the contact settles."""
        self.gripper_state = gripper_state
        qpos = self._current_qpos()
        for _ in range(self.settle_steps):
            self._step_with_qpos(qpos)
            qpos = self._current_qpos()

    def open_gripper(self) -> None:
        """Open the gripper; step a few times so the release settles."""
        self.gripper_state = self.OPEN
        qpos = self._current_qpos()
        for _ in range(self.settle_steps):
            self._step_with_qpos(qpos)
            qpos = self._current_qpos()

    def hold(self, n_steps: int = 30) -> None:
        """Hold the current pose so `is_robot_static` flips true."""
        qpos = self._current_qpos()
        for _ in range(n_steps):
            self._step_with_qpos(qpos)
            qpos = self._current_qpos()


def run_scripted_episode(
    planner: "IKMotionPlanner",
    grasp_pose: sapien.Pose,
    goal_pos: np.ndarray,
) -> None:
    """Execute the 7 keyframe phases via the planner.

    approach → descend → grasp → lift → transport → place → hold.
    Returns nothing; read success via `env.unwrapped.evaluate()`.

    The final phase holds the cube at the goal position instead of
    releasing — the env's success check only requires
    `is_obj_placed AND is_robot_static`, and dropping the cube from
    even a small height tends to bounce it past the 1.56 cm goal
    threshold.

    Args:
        planner: `IKMotionPlanner` bound to a fresh-reset env.
        grasp_pose: Target TCP pose at gripper closure.
        goal_pos: `(3,)` xyz drop point in world coordinates.
    """
    quat = grasp_pose.q
    goal = np.asarray(goal_pos)

    # 1. approach (10cm above grasp pose)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.10]) * grasp_pose
    )
    # 2. descend (3cm above)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.03]) * grasp_pose
    )
    # 3. at-cube (1cm above)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.01]) * grasp_pose
    )
    planner.close_gripper(gripper_state=-0.8)
    # 4. lift (15cm above grasp)
    planner.move_to_pose_with_screw(
        sapien.Pose([0, 0, 0.15]) * grasp_pose
    )
    # 5. transport (5cm above goal — closer than mplib's 15cm because
    #    the IK + interpolation path is less reach-efficient).
    planner.move_to_pose_with_screw(
        sapien.Pose(goal + np.array([0, 0, 0.05]), quat)
    )
    # 6. place (at goal, gripper still closed).
    planner.move_to_pose_with_screw(
        sapien.Pose(goal, quat)
    )
    # 7. hold — settle so `is_robot_static` flips true.
    planner.hold(n_steps=30)


def _compute_top_down_grasp_pose(env: gym.Env) -> sapien.Pose:
    """Build a top-down grasp pose for the cube's current spawn position.

    Args:
        env: Gymnasium env wrapping `PickCubeSO100-v1`.

    Returns:
        SAPIEN pose for the gripper at grasp time (rotated so the SO-100
        closing axis aligns with the cube faces).
    """
    unwrapped = env.unwrapped
    cube_pos = unwrapped.cube.pose.sp.p
    # NOTE: closing=[1, 0, 0] assumes an axis-aligned cube; PickCubeSO100-v1
    # spawns this way. For arbitrary cube yaw, derive from the OBB (see
    # ManiSkill `motionplanning/so100/solutions/pick_cube.py`).
    base = unwrapped.agent.build_grasp_pose(
        approaching=np.array([0, 0, -1]),
        closing=np.array([1, 0, 0]),
        center=cube_pos,
    )
    # SO-100's gripper frame needs this rotation so the closing axis
    # lines up with the cube faces. Matches upstream verbatim:
    # mani_skill/examples/motionplanning/so100/solutions/pick_cube.py:44
    return base * sapien.Pose(
        q=euler2quat(-np.pi / 2, 0, np.pi / 2)
    )


def run_scripted_agent(env: gym.Env, n_episodes: int = 10) -> float:
    """Run the scripted policy `n_episodes` times; report the success rate.

    Args:
        env: Gymnasium env from `make_env`. Must be constructed with
            `control_mode="pd_joint_pos"`.
        n_episodes: Number of episodes to roll out.

    Returns:
        Fraction of episodes where the cube landed in the goal zone.
    """
    successes = 0
    for ep in range(n_episodes):
        env.reset(seed=ep)
        planner = IKMotionPlanner(env)
        try:
            grasp_pose = _compute_top_down_grasp_pose(env)
            goal_pos = env.unwrapped.goal_site.pose.sp.p
            run_scripted_episode(planner, grasp_pose, goal_pos)
            info = env.unwrapped.evaluate()
            if _episode_success(info):
                successes += 1
        finally:
            try:
                planner.close()
            except Exception:
                pass
    return successes / n_episodes
