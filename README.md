---
title: GAIA Final Agent
emoji: 🕵️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
short_description: GAIA research agent for the HF Agents Course
---

# 🕵️ GAIA Final Agent

Final project for the **Hugging Face Agents Course — Unit 4 (GAIA)**.

This Space runs a genuine research agent against the 20-question GAIA level-1
subset used by the course. Answers are submitted to the official scoring API
with **exact-match** grading.

## How the agent works

The agent is a bounded **ReAct** loop built on Gemini:

1. **Plan** — read the question and decide whether more evidence is needed.
2. **Search** — query DuckDuckGo, read the top pages, and check Wikipedia.
3. **Inspect attachments** — parse PDF / DOCX / XLSX / CSV / text / code,
   and pass images and audio to Gemini as multimodal input.
4. **Watch videos** — YouTube URLs are attached to the model and a transcript
   is fetched as a fallback.
5. **Answer** — return only the final value, normalised for exact match.

There is **no hard-coded answer table**. Every answer is produced by the agent
from the question and the evidence it gathers.

## Space secrets

Add this secret in **Settings → Variables and secrets**:

| Name | Required | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | ✅ | Google AI Studio API key (`GOOGLE_API_KEY` also works). |

Optional variables:

| Name | Default | Description |
| --- | --- | --- |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Primary model. |
| `GEMINI_FALLBACK_MODELS` | `gemini-2.5-flash-lite,gemini-2.0-flash` | Comma-separated fallbacks. |
| `MAX_SEARCH_STEPS` | `4` | Maximum search iterations per question. |
| `MAX_SEARCH_RESULTS` | `6` | Results per search. |
| `MAX_FETCHED_PAGES` | `2` | Pages read per search. |

## Usage

1. Open the Space and click **Log in with Hugging Face**.
2. (Recommended) Run **Test Only — Do Not Submit** on 2–3 questions.
3. Review the answers, then click **Run & Submit All Answers**.
4. A full run takes roughly 10–25 minutes depending on model latency.

## Local development

```bash
git clone <this-space>
cd <space>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
python app.py
```

## Deploy to Hugging Face Spaces

The easiest path is to create an empty **Gradio** Space on the Hub, then push
this folder to it.

1. Go to <https://huggingface.co/new-space>, choose **Gradio**, make it **public**.
2. Push this repository to the Space:

```bash
git init
git add .
git commit -m "GAIA final agent"

# use the URL shown on your new Space page
git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
git push hf main
```

3. In the Space, open **Settings → Variables and secrets** and add:

```text
GEMINI_API_KEY = <your Google AI Studio key>
```

4. Wait for the build to finish, then open the Space and log in with Hugging
   Face. Run a small test first, then submit all answers.

> The `README.md` front matter already sets `sdk: gradio`, `app_file: app.py`,
> `sdk_version: 5.25.2` and `hf_oauth: true`, so no manual configuration is
> needed. `requirements.txt` installs `gradio[oauth]`, which OAuth requires.

## Notes

- The course scores with **exact string match**, so the agent is prompted to
  return only the answer value (no `FINAL ANSWER` prefix, no explanations).
- The app caches the question list in `questions_cache.json` to avoid hitting
  the course API repeatedly. Delete the file to refresh it.
- This Space is public so the `agent_code` link submitted to the leaderboard is
  verifiable.
