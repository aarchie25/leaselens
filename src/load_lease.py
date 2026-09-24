from pathlib import Path
import re
import fitz  # pymupdf

def load_pages_from_md(path: Path) -> list[str]:
    text = path.read_text()
    pieces = re.split(r"--- Page \d+ of \d+ ---", text)
    return pieces[:-1]

def load_pages_from_pdf(path: Path) -> list[str]:
    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages

def load_pages(path: Path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        return load_pages_from_pdf(path)
    elif path.suffix.lower() == ".md":
        return load_pages_from_md(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .pdf or .md")

if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1])
    pages = load_pages(path)
    print(f"Loaded {len(pages)} pages from {path.name}")
