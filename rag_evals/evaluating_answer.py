import json
import os
from urllib import response

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rag_evals.test_question import TestQuestion, get_test_questions
from retrieving_and_answering.llm_answer import answer_question
import numpy as np


load_dotenv(override=True)

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "qwen/qwen3.8-27b"


class AnswerEval(BaseModel):
    accuracy: float = Field(ge=1, le=5)
    completeness: float = Field(ge=1, le=5)
    relevance: float = Field(ge=1, le=5)
    reason: str
    generated_answer: str


def judge_answer(question, reference_answer, generated_answer, keywords):
    """Ask Qwen to score one generated answer."""
    api_key = os.getenv("QWEN_API_KEY")

    if not api_key:
        raise ValueError("QWEN_API_KEY is missing from the .env file.")

    prompt = f"""
Evaluate the generated answer against the reference answer.
Return only JSON with these fields:
accuracy, completeness, relevance, reason

Each score must be a number from 1 to 5.
Keep the reason under 30 words.

Question: {question}
Reference answer: {reference_answer}
Expected keywords: {', '.join(keywords)}
Generated answer: {generated_answer}
"""

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_completion_tokens": 150,
            "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
        },
        timeout=120,
    )

    if not response.ok:
        print(response.status_code, response.text)
        print({k: v for k, v in response.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"})
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def evaluate_answer(test: TestQuestion) -> AnswerEval:
    generated_answer, _ = answer_question(
        test.question,
        test.conversation_history,
    )

    scores = judge_answer(
        test.question,
        test.reference_answer,
        generated_answer,
        test.keywords,
    )

    return AnswerEval(
        **scores,
        generated_answer=generated_answer,
    )


def evaluate_all_answers():
    tests = [
        test
        for test in get_test_questions()
        if test.category != "unanswerable"
    ]
    tests = np.random.default_rng(21).choice(tests, size=40, replace=False)
    total_tests = len(tests)

    for index, test in enumerate(tests):
        result = evaluate_answer(test)
        progress = (index + 1) / total_tests
        yield test, result, progress
