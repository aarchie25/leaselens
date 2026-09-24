from pathlib import Path
import time
from groq import Groq

from load_lease import load_pages

LEASE_PATH = Path("data/private/lease_redacted.md")

SYNONYMS = {
    "end": ["end", "expir", "terminat", "last", "final"],
    "start": ["start", "begin", "commenc"],
    "pet": ["pet", "animal", "dog", "cat"],
    "rent": ["rent", "payment", "monthly"],
    "deposit": ["deposit", "security"],
    "leave": ["leave", "terminat", "vacate", "move out", "early"],
    "late": ["late", "overdue", "past due"],
    "quiet": ["quiet", "noise"],
    "sublet": ["sublet", "assign", "airbnb", "rental platform"],
    "notice": ["notice", "notify", "written"],
}

def expand_query(question: str) -> set[str]:
    words = set(question.lower().split())
    expanded = set(words)
    for word in words:
        for key, synonyms in SYNONYMS.items():
            if key in word:
                expanded.update(synonyms)
    return expanded

def find_relevant_pages(question: str, pages: list[str], n: int = 3) -> list[str]:
    words = expand_query(question)
    scores = []
    for page in pages:
        page_lower = page.lower()
        score = sum(1 for w in words if w in page_lower)
        scores.append(score)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [pages[i] for i in sorted(top)]

def ask(question: str, pages: list[str], client: Groq) -> str:
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
                        "content": f"""You are a helpful assistant that answers questions about a lease agreement.
Answer based only on the lease text provided. If the answer is not there, say so clearly.

LEASE TEXT:
{lease_text}

QUESTION: {question}

Give a concise, plain-English answer."""
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  [rate limited, retrying in {wait}s...]")
            time.sleep(wait)
    return "ERROR: could not get an answer, try again."

if __name__ == "__main__":
    client = Groq()
    pages = load_pages(LEASE_PATH)

    print("LeaseLens — ask anything about your lease.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        print(f"Answer: {ask(question, pages, client)}\n")
