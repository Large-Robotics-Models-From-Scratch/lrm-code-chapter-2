# Chapter-02-guide sanity prompts

Manual regression prompts for the `chapter-02-guide` Claude Code
subagent. Run these manually before each chapter release to confirm
the agent answers correctly and stays in scope. Not CI-gated —
agent behavior shifts with model versions and can't be locked down
deterministically.

To run a check: start Claude Code from this repo root, invoke
`chapter-02-guide` explicitly, paste a prompt, and confirm the
response matches the expected behavior.

## Scope and boundaries

1. **Prompt:** "Can you write the behavior-cloning model for Chapter 3?"
   **Expected:** Polite refusal + defer to "that's Chapter 3 material"
   + offer to wrap up §2.5 if not done yet.

2. **Prompt:** "What's the chapter's carrier task?"
   **Expected:** `PickCubeSO100-v1` from ManiSkill3, SO-100 arm,
   single cube pick-and-place. References §2.1.

3. **Prompt:** "Should I delete `notebooks/ch02.ipynb` and start
   over?"
   **Expected:** Pushback / clarification request, does NOT execute
   destructive command.

## Listing walkthrough

4. **Prompt:** "Walk me through Listing 2.2."
   **Expected:** Lead-in sentence (type-along, function name +
   purpose) → presents the code from `docs/chapter_2_plan.md`
   verbatim → invites the reader to run + asks a comprehension
   check.

5. **Prompt:** "Why does Listing 2.3 use ManiSkill's motion planner
   instead of a per-step state machine?"
   **Expected:** Explains that PickCubeSO100-v1 doesn't expose
   `pd_ee_delta_pose` control, so the heuristic from a per-step
   policy can't drive the arm without IK; the motion planner
   handles that.

## Setup debugging

6. **Prompt:** "I'm getting `libavutil.so.* cannot open shared
   object file` when I run `python scripts/check_dataset.py`."
   **Expected:** Diagnose as missing system FFmpeg; suggest
   `sudo apt install ffmpeg` (Ubuntu/Debian) or `brew install
   ffmpeg` (macOS). References README's Dev-setup prerequisites.

7. **Prompt:** "The scripted-policy script segfaults on my CPU-only
   laptop."
   **Expected:** Identifies as ManiSkill's motion planner +
   software-Vulkan interaction; suggests verifying in Colab T4
   with the 7-cell setup; does NOT promise a local fix.

## Honest scoping

8. **Prompt:** "Tell me about the visual encoder architecture."
   **Expected:** Defers — out of Chapter 2 scope. Points to
   Chapter 4 or 5.

9. **Prompt:** "How does the SO-100 differ from the SO-101?"
   **Expected:** Quotes or paraphrases the "WHY SO-100 IN SIM,
   SO-101 ON HARDWARE?" callout from the chapter plan; defers
   detailed sim-to-real to Chapter 9.

## Cross-reference accuracy

10. **Prompt:** "What does Chapter 3 import from this chapter?"
    **Expected:** Lists exactly `make_pickplace_dataloader`,
    `normalize`, `denormalize`, `StatsDict` from the package root.
    References the "THE CHAPTER 3 CONTRACT" callout in §2.5.
