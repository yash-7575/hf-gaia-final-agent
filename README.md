---
title: GAIA Final Agent
emoji: 🕵️
colorFrom: indigo
colorTo: blue
sdk: docker
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

The agent is a bounded **ReAct** loop:

1. **Plan** — read the question and decide whether more evidence is needed.
2. **Search** — the model runs live web searches (Groq's built-in
   `browser_search`), with a DuckDuckGo/Wikipedia fallback in the loop.
3. **Inspect attachments** — parse PDF / DOCX / XLSX / CSV / text / code;
   audio is transcribed (Whisper on Groq, native multimodal on Gemini).
4. **Watch videos** — YouTube transcripts are fetched as a fallback.
5. **Answer** — return only the final value, normalised for exact match.

There is **no hard-coded answer table**. Every answer is produced by the agent
from the question and the evidence it gathers.

## Model providers

The app picks a provider automatically:

| Priority | Provider | Requirements |
| --- | --- | --- |
| 1 | **Groq** — `openai/gpt-oss-120b` with built-in browser search | `GROQ_API_KEY` |
| 2 | **Gemini** — Google Search grounding + multimodal | `GEMINI_API_KEY` |

Groq is recommended: it is fast, free-tier friendly, and its GPT-OSS models
include live browser search.

## Space secrets

Add one of these in **Settings → Variables and secrets**:

| Name | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | ✅ (recommended) | From <https://console.groq.com/keys>. |
| `GEMINI_API_KEY` | optional | Google AI Studio key (`GOOGLE_API_KEY` also works). |

Optional variables:

| Name | Default | Description |
| --- | --- | --- |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Primary Groq model. |
| `GROQ_FALLBACK_MODELS` | `openai/gpt-oss-120b,openai/gpt-oss-20b` | Fallbacks. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Primary Gemini model. |
| `MAX_SEARCH_STEPS` | `4` | Maximum search iterations per question. |
| `MAX_SEARCH_RESULTS` | `6` | Results per search. |
| `MAX_FETCHED_PAGES` | `2` | Pages read per search. |

## Usage

1. Open the Space and click **Log in with Hugging Face**.
2. (Recommended) Run **Test Only — Do Not Submit** on 2–3 questions.
3. Review the answers, then click **Run & Submit All Answers**.
4. A full run takes roughly 10–25 minutes depending on model latency.

## Generate a GAIA leaderboard JSONL

`make_jsonl.py` runs the agent and writes the
`{"task_id", "model_answer", "reasoning_trace"}` file used by the official
GAIA leaderboard:

```bash
export GROQ_API_KEY=...
python make_jsonl.py            # all 20 questions -> gaia_submission.jsonl
python make_jsonl.py 3          # quick test on 3 questions
```

## Local development

```bash
git clone <this-space>
cd <space>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=your_key
python app.py
```

Or with Docker:

```bash
docker build -t gaia-agent .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key gaia-agent
```

## Deployment notes

This repo is deployed as a **Docker Space** (`sdk: docker`); the `Dockerfile`
installs `requirements.txt` and runs `app.py` on port 7860. `requirements.txt`
installs `gradio[oauth]`, which the Hugging Face login button requires.

Upload changes with:

```bash
hf upload <username>/<space> . --repo-type space \
  --exclude ".git/**" --exclude ".venv312/**" --exclude "__pycache__/**"
```

## Notes

- The course scores with **exact string match**, so the agent is prompted to
  return only the answer value (no `FINAL ANSWER` prefix, no explanations).
- The app caches the question list in `questions_cache.json` to avoid hitting
  the course API repeatedly. Delete the file to refresh it.
- The Space is public so the `agent_code` link submitted to the leaderboard is
  verifiable.
