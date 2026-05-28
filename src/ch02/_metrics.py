"""Internal metric helpers shared across ch02 modules."""


def _episode_success(info: dict) -> bool:
    """Coerce `info["success"]` (a (1,) torch.bool) to a Python bool.

    Args:
        info: ManiSkill step's info dict.

    Returns:
        True iff the episode reported success.
    """
    # TODO: return np.ndarray if env is ever built with num_envs > 1.
    return bool(info.get("success", False))
