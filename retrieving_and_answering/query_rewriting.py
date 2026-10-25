import re


TERM_EXPANSIONS = {
    "reval": "re-evaluation",
    "cgpa": "cumulative grade point average",
    "gpa": "grade point average",
    "exam": "examination",
}

TOPICS = {
    "attendance relaxation": r"\battendance relaxation\b",
    "attendance": r"\battendance\b",
    "re-evaluation": r"\bre[ -]?evaluation\b|\breval\b",
    "examination": r"\bexam(?:ination)?s?\b",
    "CGPA": r"\bcgpa\b",
    "scholarship reminder": r"\bscholarship(?: submission)? reminder\b",
    "scholarship": r"\bscholarships?\b",
    "fees": r"\bfees?\b|\brefunds?\b",
    "hostel": r"\bhostel\b",
    "library": r"\blibrary\b",
    "internship": r"\binternships?\b",
    "placement": r"\bplacements?\b",
}

REFERENCE_WORDS = r"\b(it|its|that|this|they|them|those)\b"

GENERIC_FOLLOW_UP = re.compile(
    r"^(?:do you have (?:any )?more details"
    r"|is there (?:any )?more details(?: regarding (?:that|it))?"
    r"|tell me more"
    r"|can you (?:elaborate|give (?:me )?more details))\??$",
    re.I,
)

def _message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return ""


def _topics(text):
    return [topic for topic, pattern in TOPICS.items() if re.search(pattern, text, re.I)]


def rewrite_query(question, history=None):
    """Expand known terms and resolve a clear reference to the previous user topic."""
    rewritten = question.strip()

    for short, long in TERM_EXPANSIONS.items():
        rewritten = re.sub(
            rf"\b{short}\b",
            lambda match: f"{match.group(0)} ({long})",
            rewritten,
            flags=re.I,
        )

    if GENERIC_FOLLOW_UP.fullmatch(question.strip()) and not _topics(question):
        previous_user = next(
            (
                message for message in reversed(history or [])
                if message.get("role") == "user"
                and _topics(_message_text(message.get("content", "")))
            ),
            None,
        )
        if previous_user:
            previous_question = _message_text(previous_user["content"])
            return f"{previous_question} {rewritten}"

    if not re.search(REFERENCE_WORDS, question, re.I):
        return rewritten

    previous_user = next(
        (message for message in reversed(history or []) if message.get("role") == "user"),
        None,
    )
    if not previous_user:
        return rewritten

    previous_topics = _topics(_message_text(previous_user.get("content", "")))
    current_topics = _topics(question)
    if not previous_topics or current_topics:
        return rewritten

    return re.sub(REFERENCE_WORDS, previous_topics[0], rewritten, count=1, flags=re.I)


def should_rewrite(question, history=None):
    """Retain the previous function signature for existing callers."""
    return rewrite_query(question, history) != question.strip()
