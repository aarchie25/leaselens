from pathlib import Path
import time
from flask import Flask, request, jsonify, render_template_string
from groq import Groq
from load_lease import load_pages

app = Flask(__name__)
client = Groq()

UPLOAD_FOLDER = Path("data/private/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

pages_cache = {}

SYNONYMS = {
    "end": ["end", "expir", "terminat", "last", "final", "december"],
    "start": ["start", "begin", "commenc"],
    "pet": ["pet", "animal", "animals", "dog", "cat", "reptile", "bird", "mammal"],
    "rent": ["rent", "payment", "monthly", "charge"],
    "deposit": ["deposit", "security", "refundable"],
    "leave": ["leave", "terminat", "vacate", "early", "move out"],
    "late": ["late", "overdue", "past due"],
    "quiet": ["quiet", "noise", "hours"],
    "sublet": ["sublet", "assign", "airbnb", "platform"],
    "notice": ["notice", "notify", "written"],
    "lease": ["lease", "term", "agreement", "expir"],
    "fee": ["fee", "charge", "cost", "fine"],
    "smoke": ["smoke", "smoking", "tobacco"],
    "parking": ["parking", "garage", "vehicle"],
    "pool": ["pool", "spa", "swimming"],
}

ALWAYS_INCLUDE = [0, 1, 2]

def expand_query(question):
    words = set(question.lower().split())
    expanded = set(words)
    for word in words:
        for key, synonyms in SYNONYMS.items():
            if key in word:
                expanded.update(synonyms)
    return expanded

def find_relevant_pages(question, pages, n=5):
    words = expand_query(question)
    scores = []
    for i, page in enumerate(pages):
        score = sum(1 for w in words if w in page.lower())
        if i in ALWAYS_INCLUDE:
            score += 5
        scores.append(score)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [pages[i] for i in sorted(top)]

HTML = open(Path(__file__).parent / "templates" / "index.html").read()

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"})
    save_path = UPLOAD_FOLDER / f.filename
    f.save(str(save_path))
    try:
        pages = load_pages(save_path)
        pages_cache[f.filename] = pages
        return jsonify({"filename": f.filename, "pages": len(pages)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    filename = data.get("filename", "")
    if filename not in pages_cache:
        return jsonify({"error": "Please upload a lease file first."})
    pages = pages_cache[filename]
    relevant = find_relevant_pages(question, pages)
    lease_text = "\n\n".join(relevant)
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                max_tokens=300,
                messages=[{"role": "user", "content": f"Answer this lease question based only on the text below.\n\nLEASE TEXT:\n{lease_text}\n\nQUESTION: {question}\n\nGive a concise plain-English answer."}]
            )
            return jsonify({"answer": response.choices[0].message.content})
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    return jsonify({"error": "Could not get an answer, please try again."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
