"""GAIA Final Agent — Hugging Face Agents Course, Unit 4.

A genuine research agent (no hard-coded answers):
  * searches the web and reads pages,
  * parses attached files (PDF, DOCX, XLSX, CSV, code, images, audio),
  * watches YouTube videos / transcripts,
  * and reasons in a bounded ReAct loop before returning an exact-match answer.

The Space exposes a Gradio UI that fetches the course questions, runs the agent
and submits the answers to the official scoring endpoint.
"""

import io
import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
QUESTIONS_CACHE_FILE = "questions_cache.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODELS = list(
    dict.fromkeys(
        [GEMINI_MODEL]
        + [
            m.strip()
            for m in os.getenv(
                "GEMINI_FALLBACK_MODELS",
                "gemini-2.5-flash-lite,gemini-2.0-flash",
            ).split(",")
            if m.strip()
        ]
    )
)

MAX_SEARCH_STEPS = int(os.getenv("MAX_SEARCH_STEPS", "4"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "6"))
MAX_FETCHED_PAGES = int(os.getenv("MAX_FETCHED_PAGES", "2"))
PAGE_CHARS = int(os.getenv("PAGE_CHARS", "6000"))
FILE_CHARS = int(os.getenv("FILE_CHARS", "20000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
SEARCH_HTTP_TIMEOUT = int(os.getenv("SEARCH_HTTP_TIMEOUT", "10"))
SEARCH_GROUNDING = os.getenv("DISABLE_SEARCH_GROUNDING", "0") != "1"
REQUEST_TIMEOUT = 20
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
YOUTUBE_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([\w-]{6,})",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def clean_answer(text: str) -> str:
    """Normalise a model answer into the bare value expected by exact match."""
    if not text:
        return ""

    text = str(text).strip()
    text = re.sub(r"(?is)^```[a-zA-Z0-9]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Keep the meaningful line: models sometimes add a sentence of explanation.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        for line in lines:
            if not re.match(r"(?i)^(thought|reasoning|explanation|note)\s*:", line):
                text = line
                break
        else:
            text = lines[-1]

    text = re.sub(r"(?is)^(final\s+answer|answer)\s*[:\-]\s*", "", text)
    text = re.sub(r"(?is)^the\s+answer\s+is\s*[:\-]?\s*", "", text)
    text = re.sub(r"\[\d+\]", "", text)  # search-grounding citation markers
    text = text.strip().strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _is_rate_limit(error: Exception) -> bool:
    msg = str(error).lower()
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "rate limit" in msg
        or "quota exceeded" in msg
    )


def _is_auth_error(error: Exception) -> bool:
    msg = str(error).lower()
    return (
        "api key not valid" in msg
        or "unauthenticated" in msg
        or "permission denied" in msg
    )


def _short_error(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error).strip()[:200]}"


# --------------------------------------------------------------------------- #
# Web tools
# --------------------------------------------------------------------------- #
def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """DuckDuckGo search (lite endpoint first) with resilient fallbacks."""
    query = (query or "").strip()
    if not query:
        return []

    endpoints = (
        f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}",
        f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
    )
    for endpoint in endpoints:
        try:
            response = requests.get(
                endpoint, headers={"User-Agent": USER_AGENT}, timeout=SEARCH_HTTP_TIMEOUT
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            nodes = soup.select(".result") or soup.select("a.result-link")
            for node in nodes:
                title_tag = node.select_one(".result__a") or node
                href = title_tag.get("href", "")
                if "uddg=" in href:
                    href = parse_qs(urlparse(href).query).get("uddg", [href])[0]
                snippet_tag = node.select_one(".result__snippet")
                results.append(
                    {
                        "title": title_tag.get_text(" ", strip=True),
                        "url": href,
                        "snippet": snippet_tag.get_text(" ", strip=True) if snippet_tag else "",
                    }
                )
                if len(results) >= max_results:
                    break

            if not results:  # lite endpoint uses a flat link list
                for node in soup.select("td a[href^='http']"):
                    href = node.get("href", "")
                    if "uddg=" in href:
                        href = parse_qs(urlparse(href).query).get("uddg", [href])[0]
                    if href.startswith("http"):
                        results.append(
                            {"title": node.get_text(" ", strip=True), "url": href, "snippet": ""}
                        )
                    if len(results) >= max_results:
                        break

            if results:
                return results
        except Exception as exc:  # noqa: BLE001
            print(f"[search] {urlparse(endpoint).hostname} failed: {_short_error(exc)}")

    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results, backend="duckduckgo"))
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("href") or item.get("url") or "",
                "snippet": item.get("body", ""),
            }
            for item in raw
            if item.get("href") or item.get("url")
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"[search] ddgs fallback failed: {_short_error(exc)}")
        return []


def fetch_page_text(url: str, max_chars: int = PAGE_CHARS) -> str:
    """Download a page (HTML or PDF) and return readable text."""
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(response.content))
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
                return re.sub(r"\s+", " ", text)[:max_chars]
            except Exception:  # noqa: BLE001
                return ""

        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text)[:max_chars]
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch] {url[:60]} failed: {_short_error(exc)}")
        return ""


def wikipedia_summary(query: str) -> str:
    """Best-effort Wikipedia lookup, useful for entity questions."""
    try:
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=SEARCH_HTTP_TIMEOUT,
        )
        search.raise_for_status()
        hits = search.json().get("query", {}).get("search", [])
        if not hits:
            return ""
        title = hits[0]["title"]
        summary = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(title)}",
            headers={"User-Agent": USER_AGENT},
            timeout=SEARCH_HTTP_TIMEOUT,
        )
        summary.raise_for_status()
        data = summary.json()
        extract = data.get("extract", "")
        return f"Wikipedia — {data.get('title', title)}:\n{extract[:2000]}" if extract else ""
    except Exception:  # noqa: BLE001
        return ""


def gather_search(query: str) -> str:
    """Run a search and enrich it with page text + a Wikipedia summary."""
    results = web_search(query)
    if not results:
        return f"No web results found for: {query}"

    chunks = [f"SEARCH QUERY: {query}"]
    for i, item in enumerate(results, 1):
        chunks.append(
            f"[{i}] {item['title']}\n"
            f"URL: {item['url']}\n"
            f"Snippet: {item['snippet']}"
        )

    for item in results[:MAX_FETCHED_PAGES]:
        page = fetch_page_text(item["url"])
        if page:
            chunks.append(f"[Page text from {item['url']}]\n{page}")

    wiki = wikipedia_summary(query)
    if wiki:
        chunks.append(wiki)

    return "\n\n".join(chunks)


def youtube_transcript(url: str) -> str:
    """Fetch a YouTube transcript when the package is available."""
    match = YOUTUBE_RE.search(url)
    if not match:
        return ""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        video_id = match.group(1)
        try:
            fetched = YouTubeTranscriptApi().fetch(video_id)
            pieces = [seg.text for seg in fetched]
        except AttributeError:  # older library versions
            fetched = YouTubeTranscriptApi.get_transcript(video_id)
            pieces = [seg["text"] if isinstance(seg, dict) else str(seg) for seg in fetched]
        text = " ".join(pieces)
        return re.sub(r"\s+", " ", text)[:15000]
    except Exception as exc:  # noqa: BLE001
        print(f"[youtube] transcript failed: {_short_error(exc)}")
        return ""


# --------------------------------------------------------------------------- #
# File tools
# --------------------------------------------------------------------------- #
def download_task_file(task_id: str):
    """Download the file attached to a task, if any."""
    if not task_id:
        return None, None, None
    try:
        response = requests.get(
            f"{DEFAULT_API_URL}/files/{task_id}", timeout=REQUEST_TIMEOUT + 10
        )
        if response.status_code == 404:
            return None, None, None
        response.raise_for_status()

        content = response.content
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        disposition = response.headers.get("content-disposition", "")
        match = re.search(r'filename="?([^"]+)"?', disposition)
        filename = match.group(1) if match else task_id

        guessed, _ = mimetypes.guess_type(filename)
        return filename, content, (content_type or guessed or "application/octet-stream")
    except Exception as exc:  # noqa: BLE001
        print(f"[file] download failed for {task_id}: {_short_error(exc)}")
        return None, None, None


def run_provided_python(code: str) -> str:
    """Execute a benchmark-provided Python file in an isolated subprocess.

    Only the downloaded file is executed (never model-generated code), in a
    throw-away directory with a scrubbed environment and a hard timeout.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "task_script.py")
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(code)
            env = {
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                "PYTHONIOENCODING": "utf-8",
                "HOME": tmp,
            }
            result = subprocess.run(
                [sys.executable, "-I", script],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
                check=False,
            )
            output = (result.stdout or "").strip()
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr.strip()}"
            return output[:4000]
    except subprocess.TimeoutExpired:
        return "[execution timed out after 15s]"
    except Exception as exc:  # noqa: BLE001
        return f"[execution failed: {_short_error(exc)}]"


def extract_file_text(filename: str, content: bytes, mime_type: str) -> str:
    """Extract plain text from a file, when the format allows it."""
    if not content:
        return ""
    name = (filename or "").lower()
    mime = (mime_type or "").lower()

    try:
        if mime.startswith("text/") or name.endswith(
            (".txt", ".md", ".csv", ".json", ".xml", ".html", ".py", ".js", ".ts", ".log")
        ):
            return content.decode("utf-8", errors="replace")[:FILE_CHARS]

        if name.endswith(".pdf") or "pdf" in mime:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages = [(page.extract_text() or "") for page in reader.pages]
            return "\n".join(pages)[:FILE_CHARS]

        if name.endswith(".docx"):
            import docx

            document = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in document.paragraphs)[:FILE_CHARS]

        if name.endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
            chunks = []
            for sheet_name, frame in sheets.items():
                chunks.append(f"Sheet: {sheet_name}")
                chunks.append(frame.to_string(index=False))
            return "\n\n".join(chunks)[:FILE_CHARS]
    except Exception as exc:  # noqa: BLE001
        return f"[could not extract file text: {_short_error(exc)}]"

    return ""


def build_file_context(task_id: str, file_name: str):
    """Return (text_context, multimodal_parts, notes) for an attached file."""
    if not task_id or not file_name:
        return "", [], ""

    filename, content, mime_type = download_task_file(task_id)
    if not content:
        return "", [], ""

    name = (filename or "").lower()
    mime = (mime_type or "").lower()
    parts: list[Any] = []
    notes = f"Attached file: {filename} ({mime_type}, {len(content)} bytes)"

    if mime.startswith("image/"):
        try:
            parts.append(types.Part.from_bytes(data=content, mime_type=mime_type))
            return "", parts, notes + " — inspect the image directly."
        except Exception as exc:  # noqa: BLE001
            return "", [], notes + f" — image attach failed: {_short_error(exc)}"

    if mime.startswith("audio/"):
        try:
            parts.append(types.Part.from_bytes(data=content, mime_type=mime_type))
            return "", parts, notes + " — listen to the audio directly."
        except Exception as exc:  # noqa: BLE001
            return "", [], notes + f" — audio attach failed: {_short_error(exc)}"

    if mime.startswith("video/") and len(content) < 15 * 1024 * 1024:
        try:
            parts.append(types.Part.from_bytes(data=content, mime_type=mime_type))
            return "", parts, notes + " — watch the video directly."
        except Exception as exc:  # noqa: BLE001
            print(f"[file] video attach failed: {_short_error(exc)}")

    text = extract_file_text(filename, content, mime_type)

    if name.endswith(".py") and text:
        execution = run_provided_python(text)
        if execution:
            text += f"\n\n--- Python execution output ---\n{execution}"

    return text, parts, notes


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


_CLIENT: Any = None


def get_client():
    global _CLIENT
    if _CLIENT is None:
        api_key = get_gemini_api_key()
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Add it as a Space secret."
            )
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def config_for_model(model_name: str, use_search: bool) -> types.GenerateContentConfig:
    """Temperature 0, bounded thinking on 2.5 models, optional search grounding."""
    kwargs: dict[str, Any] = {
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if use_search:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    if model_name.startswith("gemini-2.5"):
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "2048"))
        )
    return types.GenerateContentConfig(**kwargs)


def _call_model(client, model_name: str, contents: list[Any], use_search: bool) -> Any:
    return client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config_for_model(model_name, use_search=use_search),
    )


def generate(contents: list[Any]) -> str:
    """Call Gemini with model fallbacks and rate-limit retries.

    Every call is made with Google Search grounding enabled, so the model can
    run its own live Google searches; if a model rejects the tool the call is
    retried without it.
    """
    client = get_client()

    last_error: Exception | None = None
    for model_name in GEMINI_FALLBACK_MODELS:
        for attempt in range(3):
            try:
                try:
                    response = _call_model(client, model_name, contents, use_search=SEARCH_GROUNDING)
                except Exception as exc:  # noqa: BLE001
                    if SEARCH_GROUNDING and not _is_rate_limit(exc):
                        print(
                            f"[gemini] {model_name} rejected search grounding; retrying without: "
                            f"{_short_error(exc)}"
                        )
                        response = _call_model(
                            client, model_name, contents, use_search=False
                        )
                    else:
                        raise

                text = (getattr(response, "text", "") or "").strip()
                if text:
                    return text
                last_error = RuntimeError(f"empty response from {model_name}")
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if _is_auth_error(exc):
                    raise RuntimeError(
                        "GEMINI_API_KEY was rejected (401). Use a Google AI Studio key "
                        "from https://aistudio.google.com/apikey — it starts with 'AIza'."
                    ) from exc
                if _is_rate_limit(exc):
                    wait = 20 * (attempt + 1)
                    print(f"[gemini] rate limited on {model_name}; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                print(f"[gemini] {model_name} failed: {_short_error(exc)}")
                break

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini call failed for an unknown reason")


def parse_action(text: str):
    """Parse the ReAct response into ('search', query) or ('answer', value)."""
    raw = (text or "").strip()

    action_match = re.search(r"(?im)^\s*ACTION\s*:\s*(SEARCH|ANSWER)\b", raw)
    if action_match:
        action = action_match.group(1).upper()
        if action == "SEARCH":
            query_match = re.search(r"(?is)^\s*QUERY\s*:\s*(.+)$", raw[action_match.end():])
            if query_match:
                return "search", query_match.group(1).strip().strip("\"'")
            # Model may have written "ACTION: SEARCH: query" on one line.
            tail = raw[action_match.end():].strip().lstrip(":-").strip()
            return ("search", tail) if tail else ("search", None)

        answer_match = re.search(r"(?is)^\s*ANSWER\s*:\s*(.+)$", raw[action_match.end():])
        if answer_match:
            return "answer", answer_match.group(1).strip()
        tail = raw[action_match.end():].strip().lstrip(":-").strip()
        return ("answer", tail) if tail else ("answer", "")

    # Tolerate shorthand like "SEARCH: ..." or a bare final answer.
    shorthand = re.match(r"(?is)^\s*SEARCH\s*[:\-]\s*(.+)$", raw)
    if shorthand:
        return "search", shorthand.group(1).strip()

    return "answer", raw


def build_prompt(question: str, observations: list[str], step: int, file_note: str) -> str:
    evidence = "\n\n".join(observations) if observations else "No evidence gathered yet."
    remaining = MAX_SEARCH_STEPS - step
    return f"""You are an expert research agent solving a question from the GAIA benchmark.
Answers are graded by EXACT string match, so precision is critical.

You can search the web. Reply in EXACTLY ONE of these two formats:

To search:
THOUGHT: <one short sentence>
ACTION: SEARCH
QUERY: <a concise, specific search query>

To finish:
THOUGHT: <one short sentence>
ACTION: ANSWER
ANSWER: <the final answer ONLY>

Rules for the final answer:
- Output ONLY the answer value — no explanation, no "Answer:", no "FINAL ANSWER".
- Do not wrap the answer in quotes.
- Use the exact spelling, capitalisation and punctuation the question implies.
- If a number is requested, output just the number (keep decimals when required).
- If a list is requested, use a comma+space separated list in the requested order.
- Base every claim on the evidence below; never invent facts.

{file_note}

QUESTION:
{question}

EVIDENCE GATHERED SO FAR:
{evidence}

You have {remaining} search step(s) left. Search again if the evidence is not enough,
otherwise give the final answer.
"""


def build_force_answer_prompt(question: str, observations: list[str], file_note: str) -> str:
    evidence = "\n\n".join(observations) if observations else "No evidence gathered."
    return f"""You must now give your best final answer to the question below.
Return ONLY the answer value — no explanation, no labels, no quotes.
Use exact spelling, capitalisation, punctuation and formatting.

{file_note}

QUESTION:
{question}

EVIDENCE:
{evidence}
""".strip()


class GaiaAgent:
    """A bounded ReAct research agent built on Gemini + web tools."""

    def __init__(self, delay_seconds: float = 1.0):
        self.delay_seconds = delay_seconds

    def __call__(self, question: str, task_id: str | None = None, file_name: str | None = None) -> str:
        print("\n" + "-" * 70)
        print(f"[agent] {question[:160]}")

        file_text, file_parts, file_note = build_file_context(task_id, file_name or "")
        observations: list[str] = []
        if file_text:
            observations.append(f"ATTACHED FILE CONTENT:\n{file_text}")

        # YouTube links: give the model the video itself and a transcript fallback.
        youtube_parts: list[Any] = []
        for url in YOUTUBE_RE.findall(question):
            full = f"https://www.youtube.com/watch?v={url}"
            try:
                youtube_parts.append(
                    types.Part(file_data=types.FileData(file_uri=full))
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[youtube] attach failed: {_short_error(exc)}")
            transcript = youtube_transcript(full)
            if transcript:
                observations.append(f"YOUTUBE TRANSCRIPT ({full}):\n{transcript}")

        all_parts = file_parts + youtube_parts

        try:
            for step in range(MAX_SEARCH_STEPS):
                prompt = build_prompt(question, observations, step, file_note)
                raw = generate([prompt] + all_parts)
                action, payload = parse_action(raw)
                print(f"[agent] step {step + 1}: {action} -> {str(payload)[:100]}")

                if action == "answer":
                    answer = clean_answer(payload)
                    if answer:
                        print(f"[agent] answer: {answer}")
                        return answer
                    break

                if not payload:
                    observations.append("The last search had no query; try a different query.")
                    continue

                observations.append(gather_search(payload))
                time.sleep(self.delay_seconds)

            final_raw = generate(
                [build_force_answer_prompt(question, observations, file_note)] + all_parts
            )
            answer = clean_answer(parse_action(final_raw)[1])
            print(f"[agent] final answer: {answer}")
            return answer

        except Exception as exc:  # noqa: BLE001
            print(f"[agent] failed: {_short_error(exc)}")
            if observations:
                return "Unable to determine"
            raise


# --------------------------------------------------------------------------- #
# Course API
# --------------------------------------------------------------------------- #
def fetch_questions() -> list[dict]:
    if os.path.exists(QUESTIONS_CACHE_FILE):
        try:
            with open(QUESTIONS_CACHE_FILE, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached:
                print("[api] using cached questions")
                return cached
        except Exception:  # noqa: BLE001
            pass

    for attempt in range(3):
        response = requests.get(f"{DEFAULT_API_URL}/questions", timeout=REQUEST_TIMEOUT)
        if response.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"[api] questions rate limited; sleeping {wait}s")
            time.sleep(wait)
            continue
        response.raise_for_status()
        questions = response.json()
        with open(QUESTIONS_CACHE_FILE, "w", encoding="utf-8") as handle:
            json.dump(questions, handle, ensure_ascii=False, indent=2)
        return questions

    raise RuntimeError("Could not fetch questions: the API is rate limited.")


# --------------------------------------------------------------------------- #
# UI actions
# --------------------------------------------------------------------------- #
def _run(limit: int, submit: bool, profile, progress) -> tuple[str, pd.DataFrame]:
    if submit and not profile:
        return "Please log in with Hugging Face before submitting.", None

    if not get_gemini_api_key():
        return (
            "Missing GEMINI_API_KEY. Add it under Settings → Variables and secrets "
            "in this Space, then restart.",
            None,
        )

    try:
        questions = fetch_questions()
    except Exception as exc:  # noqa: BLE001
        return f"Error fetching questions: {_short_error(exc)}", None

    if not questions:
        return "The question list is empty.", None

    try:
        limit = int(limit)
    except Exception:  # noqa: BLE001
        limit = 0
    selected = questions if limit <= 0 else questions[:limit]

    agent = GaiaAgent()
    rows: list[dict] = []
    payload: list[dict] = []

    for index, item in enumerate(selected):
        task_id = item.get("task_id", "")
        question = item.get("question", "")
        file_name = item.get("file_name", "")
        progress((index) / max(len(selected), 1), desc=f"Question {index + 1}/{len(selected)}")

        try:
            answer = clean_answer(agent(question, task_id=task_id, file_name=file_name))
        except Exception as exc:  # noqa: BLE001
            answer = "Unable to determine"
            print(f"[run] task {task_id} failed: {_short_error(exc)}")

        if not answer:
            answer = "Unable to determine"

        payload.append({"task_id": task_id, "submitted_answer": answer})
        rows.append(
            {
                "Task ID": task_id,
                "Question": question,
                "File": file_name,
                "Answer": answer,
            }
        )

    frame = pd.DataFrame(rows)

    if not submit:
        return (
            f"Test run finished on {len(selected)} question(s). "
            "Nothing was submitted — review the answers above.",
            frame,
        )

    space_id = os.getenv("SPACE_ID", "")
    agent_code = (
        f"https://huggingface.co/spaces/{space_id}/tree/main"
        if space_id
        else "SPACE_ID_NOT_FOUND"
    )

    submission = {
        "username": profile.username.strip(),
        "agent_code": agent_code,
        "answers": payload,
    }

    try:
        response = requests.post(
            f"{DEFAULT_API_URL}/submit", json=submission, timeout=90
        )
        response.raise_for_status()
        result = response.json()
        status = (
            "Submission successful!\n"
            f"User: {result.get('username')}\n"
            f"Score: {result.get('score', 'N/A')}% "
            f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')} correct)\n"
            f"Message: {result.get('message', 'No message received.')}"
        )
        return status, frame
    except requests.exceptions.HTTPError as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", exc.response.text[:300])
            except Exception:  # noqa: BLE001
                detail = exc.response.text[:300]
        return f"Submission failed (HTTP {getattr(exc.response, 'status_code', '?')}): {detail}", frame
    except Exception as exc:  # noqa: BLE001
        return f"Submission failed: {_short_error(exc)}", frame


def run_evaluation(profile: gr.OAuthProfile | None, limit, progress=gr.Progress()):
    return _run(limit, submit=True, profile=profile, progress=progress)


def run_test(limit, progress=gr.Progress()):
    return _run(limit, submit=False, profile=None, progress=progress)


with gr.Blocks(title="GAIA Final Agent") as demo:
    gr.Markdown(
        """
        # 🕵️ GAIA Final Agent
        A research agent for the **Hugging Face Agents Course — Unit 4**.
        It searches the web, reads attached files (PDF, DOCX, XLSX, CSV, code,
        images, audio), watches YouTube videos, and returns exact-match answers.

        **How to use**
        1. Log in with Hugging Face (used as your leaderboard username).
        2. Optionally run a **Test** on a few questions first (nothing is submitted).
        3. Click **Run & Submit All Answers**. This can take 10–25 minutes.
        """
    )

    gr.LoginButton()

    with gr.Row():
        limit_input = gr.Number(
            label="Questions to run (0 = all 20)",
            value=0,
            precision=0,
            minimum=0,
        )

    with gr.Row():
        test_button = gr.Button("Test Only — Do Not Submit")
        submit_button = gr.Button("Run & Submit All Answers", variant="primary")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=7, interactive=False
    )
    results_table = gr.DataFrame(label="Questions and Answers", wrap=True)

    test_button.click(
        fn=run_test,
        inputs=[limit_input],
        outputs=[status_output, results_table],
    )
    submit_button.click(
        fn=run_evaluation,
        inputs=[limit_input],
        outputs=[status_output, results_table],
    )


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    if os.getenv("SPACE_ID"):
        print(f"SPACE_ID: {os.getenv('SPACE_ID')}")
        print(f"Repo: https://huggingface.co/spaces/{os.getenv('SPACE_ID')}/tree/main")
    if os.getenv("SPACE_HOST"):
        print(f"Runtime: https://{os.getenv('SPACE_HOST')}.hf.space")
    print("-" * 75 + "\n")

    demo.queue(default_concurrency_limit=1)
    demo.launch()
