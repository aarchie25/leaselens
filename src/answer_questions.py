from pathlib import Path
import re
import time
from groq import Groq

from load_lease import load_pages

LEASE_PATH = Path("data/private/lease_redacted.md")
QUESTIONS_PATH = Path("data/private/questions.md")

def load_questions(path: Path) -> list[str]:
    text = path.read_text()
    questions = re.findall(r"\*\*Q\d+\.\*\*\s+(.+)", text)
    return questions

def find_relevant_pages(question: str, pages: list[str], n: int = 3) -> list[str]:
    words = set(question.lower().split())
    scores = []
    for page in pages:
        page_lower = page.lower()
        score = sum(1 for w in words if w in page_lower)
        scores.append(score)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [pages[i] for i in sorted(top)]

def answer_question(question: str, pages: list[str], client: Groq) -> str:
    relevant = find_relevant_pages(question, pages)
    lease_text = "\n\n".join(relevant)
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Answer this lease question concisely based only on the text below.

LEASE TEXT:
{lease_text}

QUESTION: {question}"""
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  [retry {attempt+1}, waiting {wait}s]")
            time.sleep(wait)
    return "ERROR: failed after 5 attempts"

if __name__ == "__main__":
    client = Groq()
    pages = load_pages(LEASE_PATH)
    questions = load_questions(QUESTIONS_PATH)

    print(f"Loaded {len(pages)} page chunks")
    print(f"Found {len(questions)} questions\n")

    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        print(f"A{i}: {answer_question(question, pages, client)}")
        print("-" * 60)
        time.sleep(5)
