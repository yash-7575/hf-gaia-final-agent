"""Submit a JSONL of agent answers to the Agents Course scoring API.

Usage:
    python submit_answers.py \
        --username yash-7575 \
        --agent-code https://huggingface.co/spaces/yash-7575/autotrain-advanced/tree/main \
        --file gaia_submission.jsonl

The JSONL must contain `task_id` and either `model_answer` or
`submitted_answer` on each line. The score is printed when the API responds.
"""

import argparse
import json
import sys

import requests

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def load_answers(path: str) -> list[dict]:
    answers = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            answer = row.get("submitted_answer", row.get("model_answer"))
            if not task_id or answer is None:
                raise ValueError(f"Row missing task_id/answer: {row}")
            answers.append({"task_id": task_id, "submitted_answer": answer})
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--agent-code", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--api", default=DEFAULT_API_URL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    answers = load_answers(args.file)
    payload = {
        "username": args.username,
        "agent_code": args.agent_code,
        "answers": answers,
    }
    print(f"Prepared {len(answers)} answers for {args.username}")
    print(f"Code link: {args.agent_code}")

    if args.dry_run:
        print(json.dumps(payload, indent=2)[:2000])
        return

    response = requests.post(f"{args.api}/submit", json=payload, timeout=90)
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        print(f"HTTP {response.status_code}: {response.text[:500]}")
        sys.exit(1)

    if response.status_code >= 400:
        print(f"Submission failed (HTTP {response.status_code}): {data}")
        sys.exit(1)

    print(
        "Submission successful!\n"
        f"Score: {data.get('score')}% "
        f"({data.get('correct_count')}/{data.get('total_attempted')} correct)\n"
        f"Message: {data.get('message')}"
    )


if __name__ == "__main__":
    main()
