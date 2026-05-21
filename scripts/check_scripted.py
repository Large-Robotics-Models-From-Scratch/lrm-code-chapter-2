#!/usr/bin/env python
"""Reader-style end-to-end check for the scripted policy (PR 3).

Run from the repo root:

    python scripts/check_scripted.py

Constructs PickCubeSO100-v1 with the scripted-friendly modes
(state_dict observations, pd_ee_delta_pose control), then runs the
scripted_policy state machine across a few episodes and prints the
success rate. This is what the reader would do after typing
Listings 2.3 and 2.4. Exits non-zero on any failure.

State-mode + ee-delta control work without Vulkan rendering on a
CPU-only box; this script does not call env.render().
"""

import sys

N_EPISODES = 3


def main() -> int:
    print("=== PR 3 scripted-policy smoke check ===\n")

    print("[1/3] Importing ch02.env + ch02.scripted ...")
    try:
        from ch02.env import make_env
        from ch02.scripted import PHASES, run_scripted_agent
    except Exception as exc:
        print(f"  FAIL: import error: {exc}")
        return 1
    print(f"  ok — PHASES has {len(PHASES)} entries")

    print(
        "[2/3] Building scripted env "
        "(state_dict + pd_ee_delta_pose) ..."
    )
    try:
        env = make_env(
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            render_mode=None,
        )
    except Exception as exc:
        print(f"  FAIL: env construction error: {exc}")
        print("  hint: ManiSkill SAPIEN needs a Vulkan loader, even")
        print("        for state mode. Install libvulkan1 +")
        print("        mesa-vulkan-drivers if missing.")
        return 1
    print("  ok")

    print(f"[3/3] Running scripted policy for {N_EPISODES} episodes ...")
    try:
        success_rate = run_scripted_agent(env, n_episodes=N_EPISODES)
    except Exception as exc:
        print(f"  FAIL: rollout error: {exc}")
        env.close()
        return 1
    finally:
        try:
            env.close()
        except Exception:
            pass

    print(f"  success_rate over {N_EPISODES} episodes: {success_rate:.0%}")
    if success_rate == 0.0:
        print(
            "  NOTE: 0% with only a few episodes is plausible "
            "(stochastic cube spawns).\n"
            "  Re-run with more episodes or eyeball "
            "a rendered episode in Colab to confirm."
        )

    print("\n=== PASS — scripted policy ran end-to-end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
