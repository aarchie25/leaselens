# LeaseLens

Ask plain-English questions about your lease and get instant answers.

## What it does

Upload your lease and ask anything:
- "When does my lease end?"
- "What is the late fee?"
- "Can I have a pet?"
- "What happens if I leave early?"

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/aarchie25/leaselens.git
cd leaselens
```

**2. Create a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install groq pymupdf
```

**4. Get a free Groq API key**
Go to https://console.groq.com/keys and create a free account.

```bash
export GROQ_API_KEY="your-key-here"
```

## Usage

```bash
python src/ask.py path/to/your/lease.md
```

Then type any question about your lease.

## Example
