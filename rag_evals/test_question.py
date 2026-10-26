from pathlib import Path
import json
from pydantic import BaseModel, Field
import config

class TestQuestion(BaseModel):
    question : str = Field(description = "The question asked by the student")
    reference_answer : str = Field(description = "The reference answer to the question asked by the student")
    keywords : list[str] = Field(description = "The answer must contain these keywords")
    relevant_chunk_ids : list[str] = Field(description = "Chunk ids of the relevant chunks based on the question")
    category : str = Field(description = "Category of the question, in which it belongs")
    conversation_history : list[dict] = Field(default_factory=list)

TEST_FILE = config.QUESTION_SET

def get_test_questions() -> list:

    tests = []

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            tests.append(TestQuestion(**data))

    return tests