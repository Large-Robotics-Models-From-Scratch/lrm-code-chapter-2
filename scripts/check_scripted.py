#!/usr/bin/env python
"""Reader-style end-to-end check for the scripted policy (PR 3).

Run from the repo root:

    python scripts/check_scripted.py

Constructs PickCubeSO100-v1 in state mode (no rendering needed),
then runs ManiSkill's motion-planner-backed 7-phase scripted policy
for a few episodes. Prints success rate. Exits non-zero on failure.
"""

import sys

try:
    from ch02.env import make_env
    from ch02.scripted import PHASES, run_scripted_agent
except ImportError as _exc:
    print(f"FAIL: import error: {_exc}")
    print('hint: did you `pip install -e ".[dev,data,sim]"`?')
    sys.exit(1)

N_EPISODES = 3


def main() -> int:
    print("=== PR 3 scripted-policy smoke check ===\n")
    print(f"[1/3] Imports ok — PHASES has {len(PHASES)} entries")

    print("[2/3] Building env (state mode, default joint control) ...")
    try:
        env = make_env(obs_mode="state", render_mode=None)
    except Exception as exc:
        print(f"  FAIL: env construction error: {exc}")
        return 1
    print("  ok")

    print(f"[3/3] Running scripted policy for {N_EPISODES} episodes ...")
    print("  (this calls ManiSkill's motion planner per episode —")
    print("   each episode takes a few seconds)")
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
            "  NOTE: 0% with only a few episodes is possible —"
            " grasp pose is\n"
            "  hardcoded top-down, no recovery on miss. Re-run "
            "with more episodes."
        )

    print("\n=== PASS — scripted policy ran end-to-end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
