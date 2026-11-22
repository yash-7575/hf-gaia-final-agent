# Submission checklist

Use this before uploading to the Hugging Face Agents Course leaderboard.

## 1. Space setup

- [ ] Create a **public** Space with SDK **Gradio**.
- [ ] Push this folder to the Space (`git push hf main`).
- [ ] Confirm the Space builds without errors in the **Logs** tab.
- [ ] Add the `GEMINI_API_KEY` secret in **Settings → Variables and secrets**.

## 2. Verify the agent

- [ ] Open the Space and log in with Hugging Face.
- [ ] Set **Questions to run** to `2` or `3`.
- [ ] Click **Test Only — Do Not Submit** and check the answers look sane.
- [ ] Confirm the logs show searches, file parsing, and a final answer.

## 3. Submit

- [ ] Set **Questions to run** to `0` (all 20).
- [ ] Click **Run & Submit All Answers** and wait (10–25 minutes).
- [ ] Confirm the status box shows a score.
- [ ] Copy the Space URL (`https://huggingface.co/spaces/<you>/<space>`) — this is
      the `agent_code` link the app sends automatically.

## 4. Keep it verifiable

- [ ] Keep the Space **public** so reviewers can inspect the code.
- [ ] Do not add hard-coded answers or benchmark-specific shortcuts.
- [ ] If you change the agent later, re-run a small test before resubmitting.

## Notes

- A full run makes many Gemini calls. Check your Google AI Studio quota first.
- The course grades with **exact string match**. The agent is prompted to return
  only the answer value, but occasional formatting mismatches are expected.
- `questions_cache.json` is created at runtime and is git-ignored.
