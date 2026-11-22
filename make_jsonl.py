"""Generate a GAIA-leaderboard JSONL file from a real agent run.

Each line: {"task_id": ..., "model_answer": ..., "reasoning_trace": ...}

Usage:
    GROQ_API_KEY=... python make_jsonl.py            # all questions
    GROQ_API_KEY=... python make_jsonl.py 5          # first 5 (quick test)
    GROQ_API_KEY=... python make_jsonl.py 0 out.jsonl
"""

import json
import sys

import app


def format_trace(agent) -> str:
    parts = []
    for step in getattr(agent, "last_trace", []):
        action = step.get("action", "")
        detail = str(step.get("detail", "")).strip()
        if action == "read_file":
            parts.append(f"Read attached file: {detail}")
        elif action == "search":
            parts.append(f"Searched: {detail}")
        elif action == "answer":
            raw = str(step.get("raw", "")).strip()
            if raw and raw != detail:
                parts.append(f"Model output: {raw}")
            else:
                parts.append(f"Answered: {detail}")
        elif action == "final_answer":
            parts.append(f"Final answer: {detail}")
        elif action == "error":
            parts.append(f"Error: {detail}")
    return "\n".join(parts) if parts else "Answered directly from the question."


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    output = sys.argv[2] if len(sys.argv) > 2 else "gaia_submission.jsonl"

    questions = app.fetch_questions()
    if limit > 0:
        questions = questions[:limit]

    agent = app.GaiaAgent()
    with open(output, "w", encoding="utf-8") as handle:
        for index, item in enumerate(questions, 1):
            task_id = item.get("task_id", "")
            question = item.get("question", "")
            file_name = item.get("file_name", "")
            print(f"\n[{index}/{len(questions)}] {task_id}", flush=True)

            try:
                answer = app.clean_answer(agent(question, task_id, file_name))
            except Exception as exc:  # noqa: BLE001
                print(f"[make_jsonl] task {task_id} failed: {exc}")
                answer = "Unable to determine"

            record = {
                "task_id": task_id,
                "model_answer": answer or "Unable to determine",
                "reasoning_trace": format_trace(agent),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"\nWrote {len(questions)} records to {output}")


if __name__ == "__main__":
    main()
