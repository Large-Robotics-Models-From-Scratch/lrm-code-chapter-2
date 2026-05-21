#!/usr/bin/env python
"""Quick env-construction smoke check for PR 2.

Run from the repo root:

    python scripts/check_env.py

Steps 1-3 use obs_mode="state" and work without Vulkan rendering.
Step 4 attempts obs_mode="rgb" + a render() call and saves a PNG to
/tmp/so100_render.png; this needs a working Vulkan driver (hardware
or software via mesa). The script exits non-zero on any failure.
"""

import sys

RENDER_OUT = "/tmp/so100_render.png"


def main() -> int:
    print("=== PR 2 env smoke check ===\n")

    print("[1/4] Importing ch02.env ...")
    try:
        from ch02.env import make_env
    except Exception as exc:
        print(f"  FAIL: import error: {exc}")
        print("  hint: did you `pip install -e \".[dev,data,sim]\"`?")
        return 1
    print("  ok")

    print("[2/4] Constructing PickCubeSO100-v1 (obs_mode='state') ...")
    try:
        env = make_env(obs_mode="state", render_mode=None)
    except Exception as exc:
        print(f"  FAIL: env construction error: {exc}")
        print("  hint: ManiSkill needs SAPIEN; on CPU-only machines this")
        print("        may still require a software Vulkan driver "
              "(libvulkan + mesa).")
        return 1
    print("  ok")

    print("[3/4] Reset + sample an action ...")
    try:
        obs, info = env.reset(seed=0)
        action = env.action_space.sample()
        print(f"  obs type: {type(obs).__name__}")
        print(f"  action space: {env.action_space}")
        print(f"  action shape: {action.shape}")
        if action.shape != (6,):
            print(
                f"  WARNING: expected action shape (6,), "
                f"got {action.shape}"
            )
    except Exception as exc:
        print(f"  FAIL: reset/sample error: {exc}")
        env.close()
        return 1
    finally:
        try:
            env.close()
        except Exception:
            pass
    print("  ok")

    print("[4/4] Constructing RGB env and rendering a frame ...")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  FAIL: matplotlib import error: {exc}")
        return 1
    rgb_env = None
    try:
        rgb_env = make_env(obs_mode="rgb")
        rgb_env.reset(seed=0)
        img = rgb_env.render()
        # ManiSkill may return a torch Tensor; coerce to numpy for matplotlib.
        arr = img.cpu().numpy() if hasattr(img, "cpu") else img
        # Vectorized envs return (n_envs, H, W, 3); take the first scene.
        if arr.ndim == 4:
            arr = arr[0]
        print(f"  rendered shape: {arr.shape}, dtype: {arr.dtype}")
        plt.figure(figsize=(6, 6))
        plt.imshow(arr)
        plt.axis("off")
        plt.savefig(RENDER_OUT, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved to {RENDER_OUT}")
    except Exception as exc:
        print(f"  FAIL: render error: {exc}")
        print("  hint: SAPIEN can't render. If `vulkaninfo --summary`")
        print("        shows a device, the issue may be SAPIEN-specific;")
        print("        rendering still works in Colab T4. Otherwise install")
        print("        mesa-vulkan-drivers + libvulkan1.")
        return 1
    finally:
        if rgb_env is not None:
            try:
                rgb_env.close()
            except Exception:
                pass
    print("  ok")

    print(
        "\n=== PASS — env + 6-DOF action + RGB render all work ===\n"
        f"open {RENDER_OUT} to see the rendered SO-100 scene"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
