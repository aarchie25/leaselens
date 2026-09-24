import argparse
import io
import json
import os
import re
import time
import uuid
from datetime import timedelta

from flask import Flask, jsonify, render_template, request, session
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24)
app.permanent_session_lifetime = timedelta(days=7)

# ---------------------------------------------------------------------------
# CONFIG: confirmed working against Groq in your logs. Change here if needed.
# ---------------------------------------------------------------------------
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
client = OpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"),
)
# ---------------------------------------------------------------------------

# Every cache below is keyed by (session_id, filename) so different visitors
# never see or overwrite each other's uploads, even if the filename matches.
pages_cache = {}    # (sid, filename) -> list of page texts
summary_cache = {}  # (sid, filename) -> summary dict
risk_cache = {}     # (sid, filename) -> list of flags
compare_cache = {}  # (sid, lease_filename, offer_filename, offer_size) -> result dict


def get_sid():
    """One private ID per browser, stored in a signed cookie. No login needed."""
    session.permanent = True
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


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
    per = max(500, max_chars // max(len(idx), 1))  # share the budget so no page gets cut off
    return "\n\n".join(f"[Page {i + 1}]\n{pages[i][:per]}" for i in idx)


def looks_like_lease(pages):
    """Free keyword check on the first pages. No AI call, no tokens."""
    text = " ".join(pages[:6]).lower()

    def has(word):
        return re.search(r"\b" + word + r"s?\b", text) is not None

    core = any(has(w) for w in ["lease", "tenancy", "rental agreement"])
    support = sum(has(w) for w in ["tenant", "landlord", "lessor", "lessee", "resident",
                                   "premises", "rent", "security deposit", "apartment", "management"])
    return core and support >= 2


USE_EFFORT = True
LIMIT = {"msg": None}


def err_msg(default):
    return LIMIT["msg"] or default


def call_llm(prompt, max_tokens):
    """Returns (text, finish_reason). Explains usage-limit errors instead of retrying blindly."""
    global USE_EFFORT
    last = None
    for attempt in range(2):
        try:
            t0 = time.time()
            kwargs = dict(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if USE_EFFORT:
                kwargs["extra_body"] = {"reasoning_effort": "low"}
            r = client.with_options(timeout=45, max_retries=0).chat.completions.create(**kwargs)
            choice = r.choices[0]
            LIMIT["msg"] = None
            print(f"LLM call took {time.time() - t0:.1f}s, finish={choice.finish_reason}", flush=True)
            return (choice.message.content or "").strip(), choice.finish_reason
        except Exception as e:
            last = e
            text = str(e)
            code = getattr(e, "status_code", None)
            print("LLM API error:", repr(e)[:400], flush=True)
            if code == 429:
                if "TPM" in text and attempt == 0:
                    time.sleep(8)
                    continue
                m = re.search(r"try again in ([0-9hms.]+)", text)
                wait = m.group(1) if m else "a few minutes"
                kind = "per-minute" if "TPM" in text else "daily"
                LIMIT["msg"] = f"AI {kind} usage limit reached. Try again in about {wait}."
                raise
            if USE_EFFORT and code == 400:
                USE_EFFORT = False  # provider rejected the option; retry without it
                continue
            time.sleep(2)
    raise last


@app.route("/")
def index():
    get_sid()
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    sid = get_sid()
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
    key = (sid, f.filename)
    pages_cache[key] = pages
    summary_cache.pop(key, None)
    risk_cache.pop(key, None)
    return jsonify({"pages": len(pages), "filename": f.filename,
                    "looks_like_lease": looks_like_lease(pages)})


@app.route("/ask", methods=["POST"])
def ask():
    sid = get_sid()
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    filename = data.get("filename") or ""
    key = (sid, filename)
    if key not in pages_cache:
        return jsonify({"error": "Please upload your lease first. The server may have restarted."})
    if not question:
        return jsonify({"error": "Please type a question."})

    pages = pages_cache[key]
    context = build_context(pages, retrieve(pages, question, k=4), max_chars=12000)
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
        text, finish = call_llm(prompt, 2000)
    except Exception:
        return jsonify({"error": err_msg("The AI service did not respond. Check the server terminal for details.")})
    if not text:
        return jsonify({"error": "The model returned an empty answer. Try rephrasing your question."})
    if finish == "length":
        text += "\n\n*(Answer was cut off. Try a more specific question.)*"
    return jsonify({"answer": text})


SUMMARY_KEYS = ["rent", "deposit", "start_date", "end_date", "late_fee", "notice"]


@app.route("/summarize", methods=["POST"])
def summarize():
    sid = get_sid()
    data = request.get_json(silent=True) or {}
    filename = data.get("filename") or ""
    key = (sid, filename)
    if key not in pages_cache:
        return jsonify({"error": "Please upload your lease first."})
    if key in summary_cache:
        return jsonify({"facts": summary_cache[key]})

    pages = pages_cache[key]
    idx = set(range(min(2, len(pages))))
    idx.update(retrieve(pages, "monthly rent security deposit late fee notice to vacate lease term", k=3))
    context = build_context(pages, sorted(idx), max_chars=15000)
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
            text, _ = call_llm(prompt, 1500)
            start, end = text.find("{"), text.rfind("}")
            raw = json.loads(text[start:end + 1])
            facts = {k: str(raw.get(k, "Not specified")) for k in SUMMARY_KEYS}
            summary_cache[key] = facts
            return jsonify({"facts": facts})
        except json.JSONDecodeError as e:
            print("Summary JSON parse failed:", e, flush=True)
        except Exception:
            break
    return jsonify({"error": err_msg("Could not generate the summary. Check the server terminal.")})


RISK_LEVELS = {"high": 0, "medium": 1, "low": 2}


@app.route("/risks", methods=["POST"])
def risks():
    sid = get_sid()
    data = request.get_json(silent=True) or {}
    filename = data.get("filename") or ""
    key = (sid, filename)
    if key not in pages_cache:
        return jsonify({"error": "Please upload your lease first."})
    if key in risk_cache:
        return jsonify({"flags": risk_cache[key]})

    pages = pages_cache[key]
    query = ("early termination liquidated damages fee penalty late fee default eviction forfeit "
             "indemnify waive arbitration automatic renewal holdover entry access attorney fees "
             "charges deductions non-refundable")
    context = build_context(pages, retrieve(pages, query, k=4), max_chars=12000)
    prompt = f"""You are reviewing a residential lease for the tenant. Using ONLY the excerpts below, list clauses that could cost the tenant money or limit their rights.
Look for: early-termination or liquidation fees, high or compounding late fees, automatic renewal or holdover rent, forfeited deposits or non-refundable fees, broad landlord entry rights, waivers of legal rights, arbitration or jury waivers, tenant paying the landlord's attorney fees, and tenant liability for other people's actions.
Return ONLY a JSON array, no other text, with at most 8 objects, most serious first:
[{{"severity": "high", "title": "short name", "detail": "1-2 plain-English sentences that include any dollar amounts or day counts", "page": 3}}]
Rules: severity is "high", "medium" or "low". Only include items actually present in the excerpts. "page" is the number from the [Page N] marker. Never invent amounts. If nothing is risky, return [].

LEASE EXCERPTS:
{context}"""

    for attempt in range(2):
        try:
            text, _ = call_llm(prompt, 2000)
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
            risk_cache[key] = flags
            return jsonify({"flags": flags})
        except json.JSONDecodeError as e:
            print("Risk JSON parse failed:", e, flush=True)
        except Exception as e:
            print("Risk scan error:", repr(e), flush=True)
            break
    return jsonify({"error": err_msg("Could not scan for risky clauses. Check the server terminal.")})


@app.route("/page", methods=["POST"])
def page():
    sid = get_sid()
    data = request.get_json(silent=True) or {}
    filename = data.get("filename") or ""
    key = (sid, filename)
    if key not in pages_cache:
        return jsonify({"error": "Please upload your lease first."})
    pages = pages_cache[key]
    try:
        n = int(data.get("n"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid page number."})
    if n < 1 or n > len(pages):
        return jsonify({"error": f"This file has {len(pages)} pages."})
    return jsonify({"page": n, "total": len(pages), "text": pages[n - 1]})


VERDICTS = {"better", "worse", "same", "unclear"}


@app.route("/compare", methods=["POST"])
def compare():
    sid = get_sid()
    lease_name = request.form.get("lease") or ""
    f = request.files.get("file")
    lease_key = (sid, lease_name)
    if lease_key not in pages_cache:
        return jsonify({"error": "Please upload your lease first."})
    if not f or not f.filename:
        return jsonify({"error": "Choose the renewal offer file first."})
    if not f.filename.lower().endswith((".pdf", ".md")):
        return jsonify({"error": "Please upload a .pdf or .md file."})
    try:
        offer_pages = load_pages(f)
    except Exception as e:
        print("Renewal upload error:", repr(e), flush=True)
        return jsonify({"error": "Could not read that file."})
    if not any(p.strip() for p in offer_pages):
        return jsonify({"error": "No text found. Scanned PDFs are not supported yet."})

    cache_key = (sid, lease_name, f.filename, sum(len(p) for p in offer_pages))
    if cache_key in compare_cache:
        return jsonify(compare_cache[cache_key])

    lease_pages = pages_cache[lease_key]
    query = ("monthly rent increase renewal term lease end date security deposit late fee "
             "pet rent termination fee notice to vacate fees charges utilities")
    lease_ctx = build_context(lease_pages, retrieve(lease_pages, query, k=3),
                              max_chars=8000).replace("[Page ", "[Lease page ")
    offer_ctx = build_context(offer_pages, retrieve(offer_pages, query, k=3),
                              max_chars=8000).replace("[Page ", "[Offer page ")
    facts = summary_cache.get((sid, lease_name))
    facts_txt = "\n".join(f"- {k}: {v}" for k, v in facts.items()) if facts else "(not available)"

    prompt = f"""You are helping a tenant compare their CURRENT lease with a RENEWAL OFFER, using ONLY the excerpts below.

Facts already extracted from the current lease (may be incomplete):
{facts_txt}

CURRENT LEASE EXCERPTS:
{lease_ctx}

RENEWAL OFFER EXCERPTS:
{offer_ctx}

Return ONLY a JSON object, no other text:
{{"overall": "2-3 plain-English sentences: what changes and whether it favors the tenant",
  "rows": [{{"item": "short name", "current": "what the current lease says", "renewal": "what the offer says", "verdict": "better", "note": "one short sentence on what changed"}}]}}
Rules:
- At most 8 rows, most important first. Cover, where present: monthly rent, lease term and dates, security deposit, late fee, pet terms and fees, early-termination fee, move-out notice, new or removed fees and clauses.
- verdict is from the tenant's point of view: "better", "worse", "same" or "unclear".
- Use "Not found" when the excerpts do not show a term, and set verdict to "unclear".
- Quote amounts exactly as written. Never invent numbers.
- In "current", cite lease pages like (p. 3). In "renewal", cite offer pages like (offer p. 2)."""

    for attempt in range(2):
        try:
            text, _ = call_llm(prompt, 2000)
            start, end = text.find("{"), text.rfind("}")
            raw = json.loads(text[start:end + 1])
            rows = []
            for r in (raw.get("rows") or [])[:8]:
                if not isinstance(r, dict):
                    continue
                v = str(r.get("verdict", "unclear")).lower()
                rows.append({
                    "item": str(r.get("item", "Term")),
                    "current": str(r.get("current", "Not found")),
                    "renewal": str(r.get("renewal", "Not found")),
                    "verdict": v if v in VERDICTS else "unclear",
                    "note": str(r.get("note", "")),
                })
            result = {"overall": str(raw.get("overall", "")), "rows": rows}
            compare_cache[cache_key] = result
            return jsonify(result)
        except json.JSONDecodeError as e:
            print("Compare JSON parse failed:", e, flush=True)
        except Exception as e:
            print("Compare error:", repr(e), flush=True)
            break
    return jsonify({"error": err_msg("Could not compare the documents. Check the server terminal.")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    # threaded=True lets two visitors' requests run at once, instead of one
    # person's slow AI call blocking the page for everyone else.
    app.run(debug=True, use_reloader=False, port=args.port, threaded=True)
