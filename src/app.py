import argparse
import io
import json
import os
import re
import time

from flask import Flask, jsonify, render_template, request
from openai import OpenAI

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CONFIG: if your old app.py created `client` differently (another base_url or
# API-key variable), copy YOUR old lines over this block. Nothing else changes.
# ---------------------------------------------------------------------------
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"),
)
# ---------------------------------------------------------------------------

pages_cache = {}    # filename -> list of page texts
summary_cache = {}  # filename -> summary dict

STOP = set("""the and for are but not you your can could would should what when where which who how
does did has have had with from that this there they them any all may will about into out over
under before after than then else its our his her him she""".split())

SYNONYMS = {
    "leave": ["terminate", "termination", "early", "vacate", "move out", "break", "notice"],
    "leaving": ["terminate", "termination", "early", "vacate", "move out", "break", "notice"],
    "move": ["vacate", "move out", "notice", "terminate"],
    "break": ["terminate", "termination", "early", "liquidated"],
    "cancel": ["terminate", "termination", "early", "notice"],
    "end": ["terminate", "expire", "expiration", "vacate", "notice"],
    "early": ["terminate", "termination", "liquidated", "re-let", "reletting"],
    "pet": ["animal", "pets", "dog", "cat"],
    "pets": ["animal", "pet", "dog", "cat"],
    "dog": ["pet", "animal"],
    "cat": ["pet", "animal"],
    "late": ["late fee", "late charge", "delinquent", "past due"],
    "fee": ["fees", "charge", "charges"],
    "rent": ["rental", "payment", "pay", "due"],
    "pay": ["payment", "rent", "due", "charge"],
    "deposit": ["security deposit", "refund", "deductions"],
    "park": ["parking", "vehicle", "garage", "decal"],
    "parking": ["park", "vehicle", "garage", "decal"],
    "guest": ["guests", "visitor", "occupant", "occupants"],
    "roommate": ["occupant", "sublet", "assign", "additional occupant"],
    "sublet": ["sublease", "assign", "assignment"],
    "noise": ["quiet", "disturb", "nuisance"],
    "repair": ["maintenance", "repairs", "damage", "service request"],
    "smoke": ["smoking", "tobacco"],
    "smoking": ["smoke", "tobacco"],
    "renew": ["renewal", "extend", "holdover", "month-to-month"],
    "insurance": ["renter", "renters", "liability"],
    "utilities": ["utility", "water", "electric", "trash", "internet"],
}


def load_pages(file_storage):
    name = file_storage.filename or ""
    data = file_storage.read()
    if name.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            PdfReader = None
        if PdfReader:
            reader = PdfReader(io.BytesIO(data))
            return [(p.extract_text() or "") for p in reader.pages]
        import fitz  # PyMuPDF fallback
        doc = fitz.open(stream=data, filetype="pdf")
        return [pg.get_text() for pg in doc]
    text = data.decode("utf-8", errors="ignore")
    return [text[i:i + 3000] for i in range(0, len(text), 3000)] or [""]


def retrieve(pages, question, k=6):
    words = {w for w in re.findall(r"[a-z]{3,}", question.lower()) if w not in STOP}
    terms = set(words)
    for w in words:
        terms.update(SYNONYMS.get(w, []))
    scored = []
    for i, page in enumerate(pages):
        low = page.lower()
        scored.append((sum(low.count(t) for t in terms), i))
    top = sorted(scored, key=lambda s: (-s[0], s[1]))[:k]
    idx = sorted(i for score, i in top if score > 0)
    return idx or list(range(min(k, len(pages))))


def build_context(pages, idx, max_chars=24000):
    parts = [f"[Page {i + 1}]\n{pages[i]}" for i in idx]
    return "\n\n".join(parts)[:max_chars]


def call_llm(prompt, max_tokens):
    """Returns (text, finish_reason). Retries only on API errors."""
    last = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,  # high enough for hidden reasoning + answer
                messages=[{"role": "user", "content": prompt}],
            )
            choice = r.choices[0]
            return (choice.message.content or "").strip(), choice.finish_reason
        except Exception as e:
            last = e
            print("LLM API error:", repr(e), flush=True)
            time.sleep(2 * (attempt + 1))
    raise last


@app.route("/")
def index():
    return render_template("index.html")


def looks_like_lease(pages):
    """Free keyword check on the first pages. No AI call, no tokens."""
    text = " ".join(pages[:6]).lower()

    def has(word):
        return re.search(r"\b" + word + r"s?\b", text) is not None

    core = any(has(w) for w in ["lease", "tenancy", "rental agreement"])
    support = sum(has(w) for w in ["tenant", "landlord", "lessor", "lessee", "resident",
                                   "premises", "rent", "security deposit", "apartment", "management"])
    return core and support >= 2


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file received."})
    if not f.filename.lower().endswith((".pdf", ".md")):
        return jsonify({"error": "Please upload a .pdf or .md file."})
    try:
        pages = load_pages(f)
    except Exception as e:
        print("Upload error:", repr(e), flush=True)
        return jsonify({"error": "Could not read that file."})
    if not any(p.strip() for p in pages):
        return jsonify({"error": "No text found. Scanned PDFs are not supported yet."})
    pages_cache[f.filename] = pages
    summary_cache.pop(f.filename, None)
    risk_cache.pop(f.filename, None)
    return jsonify({"pages": len(pages), "filename": f.filename,
                    "looks_like_lease": looks_like_lease(pages)})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    filename = data.get("filename") or ""
    if filename not in pages_cache:
        return jsonify({"error": "Please upload your lease first. The server may have restarted."})
    if not question:
        return jsonify({"error": "Please type a question."})

    pages = pages_cache[filename]
    context = build_context(pages, retrieve(pages, question))
    prompt = f"""You answer questions about a residential lease, using ONLY the excerpts below.
Rules:
- Answer in plain English in under 150 words.
- Use short sentences or a short bullet list. Do NOT use tables.
- Mention the page number(s) you relied on, like (p. 12).
- If the excerpts do not answer the question, say so and name what to look for in the lease.

LEASE EXCERPTS:
{context}

QUESTION: {question}"""
    try:
        text, finish = call_llm(prompt, 4000)
    except Exception:
        return jsonify({"error": "The AI service did not respond. Check the server terminal for details."})
    if not text:
        return jsonify({"error": "The model returned an empty answer. Try rephrasing your question."})
    if finish == "length":
        text += "\n\n*(Answer was cut off. Try a more specific question.)*"
    return jsonify({"answer": text})


SUMMARY_KEYS = ["rent", "deposit", "start_date", "end_date", "late_fee", "notice"]


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename") or ""
    if filename not in pages_cache:
        return jsonify({"error": "Please upload your lease first."})
    if filename in summary_cache:
        return jsonify({"facts": summary_cache[filename]})

    pages = pages_cache[filename]
    idx = set(range(min(3, len(pages))))
    idx.update(retrieve(pages, "monthly rent security deposit late fee notice to vacate lease term", k=8))
    context = build_context(pages, sorted(idx), max_chars=30000)
    prompt = f"""Extract these facts from the lease excerpts. Return ONLY a JSON object, no other text:
{{
  "rent": "total monthly rent",
  "deposit": "security deposit amount",
  "start_date": "lease start date",
  "end_date": "lease end date",
  "late_fee": "late fee amount and when it applies",
  "notice": "days of notice required to move out"
}}
If a fact is not found, use "Not specified".

LEASE EXCERPTS:
{context}"""

    for attempt in range(2):
        try:
            text, _ = call_llm(prompt, 3000)
            start, end = text.find("{"), text.rfind("}")
            raw = json.loads(text[start:end + 1])
            facts = {k: str(raw.get(k, "Not specified")) for k in SUMMARY_KEYS}
            summary_cache[filename] = facts
            return jsonify({"facts": facts})
        except json.JSONDecodeError as e:
            print("Summary JSON parse failed:", e, flush=True)
        except Exception:
            break
    return jsonify({"error": "Could not generate the summary. Check the server terminal."})


risk_cache = {}
RISK_LEVELS = {"high": 0, "medium": 1, "low": 2}


@app.route("/risks", methods=["POST"])
def risks():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename") or ""
    if filename not in pages_cache:
        return jsonify({"error": "Please upload your lease first."})
    if filename in risk_cache:
        return jsonify({"flags": risk_cache[filename]})

    pages = pages_cache[filename]
    query = ("early termination liquidated damages fee penalty late fee default eviction forfeit "
             "indemnify waive arbitration automatic renewal holdover entry access attorney fees "
             "charges deductions non-refundable")
    context = build_context(pages, retrieve(pages, query, k=6), max_chars=24000)
    prompt = f"""You are reviewing a residential lease for the tenant. Using ONLY the excerpts below, list clauses that could cost the tenant money or limit their rights.
Look for: early-termination or liquidation fees, high or compounding late fees, automatic renewal or holdover rent, forfeited deposits or non-refundable fees, broad landlord entry rights, waivers of legal rights, arbitration or jury waivers, tenant paying the landlord's attorney fees, and tenant liability for other people's actions.
Return ONLY a JSON array, no other text, with at most 8 objects, most serious first:
[{{"severity": "high", "title": "short name", "detail": "1-2 plain-English sentences that include any dollar amounts or day counts", "page": 3}}]
Rules: severity is "high", "medium" or "low". Only include items actually present in the excerpts. "page" is the number from the [Page N] marker. Never invent amounts. If nothing is risky, return [].

LEASE EXCERPTS:
{context}"""

    for attempt in range(2):
        try:
            text, _ = call_llm(prompt, 4000)
            start, end = text.find("["), text.rfind("]")
            raw = json.loads(text[start:end + 1])
            flags = []
            for item in raw[:8]:
                sev = str(item.get("severity", "medium")).lower()
                if sev not in RISK_LEVELS:
                    sev = "medium"
                page = item.get("page")
                flags.append({
                    "severity": sev,
                    "title": str(item.get("title", "Clause")),
                    "detail": str(item.get("detail", "")),
                    "page": page if isinstance(page, int) else None,
                })
            flags.sort(key=lambda f: RISK_LEVELS[f["severity"]])
            risk_cache[filename] = flags
            return jsonify({"flags": flags})
        except json.JSONDecodeError as e:
            print("Risk JSON parse failed:", e, flush=True)
        except Exception as e:
            print("Risk scan error:", repr(e), flush=True)
            break
    return jsonify({"error": "Could not scan for risky clauses. Check the server terminal."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    # use_reloader=False keeps uploaded leases in memory. Restart manually after edits.
    app.run(debug=True, use_reloader=False, port=args.port)
