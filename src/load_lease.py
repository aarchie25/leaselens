from pathlib import Path
import re

LEASE_PATH = Path("data/private/lease_redacted.md")

def load_pages(path: Path) -> list[str]:
    text = path.read_text()
    pieces = re.split(r"--- Page \d+ of \d+ ---", text)
    # Last piece is always metadata/junk, not a real page
    pages = pieces[:-1]
    return pages

if __name__ == "__main__":
    pages = load_pages(LEASE_PATH)
    print(f"Loaded {len(pages)} page chunks")
