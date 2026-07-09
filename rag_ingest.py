"""
rag_ingest.py — PDF ingestion script
=====================================
Ingests two sets of PDFs into separate ChromaDB collections:

  trading_docs    — trading textbooks (NSE TA module, chart patterns, etc.)
  annual_reports  — company annual reports dropped in the reports/ folder

Usage:
    python rag_ingest.py                 # full rebuild (both collections)
    python rag_ingest.py --reports-only  # only rebuild annual_reports (fast)

Output:  C:/agentic-ai/kite/rag_db/   (persistent vector store)

Naming convention for reports/ folder:
    reports/TECHM.pdf              -> symbol TECHM
    reports/RELIANCE_AR2024.pdf    -> symbol RELIANCE  (first part before _)
    reports/TCS/Q4FY25.pdf         -> symbol TCS       (subdirectory name)
"""

import sys
import re
from pathlib import Path

# ── Trading textbook sources ──────────────────────────────────────────────────
PDF_DIR = Path(r"C:\Users\salun\Desktop\Trading Material")

PDFS = [
    (
        "TA_wrkbk.pdf",
        "NSE Technical Analysis Module",
        "nse_ta",
    ),
    (
        "Idenitfying-Chart-Patterns.pdf",
        "Fidelity Chart Patterns Guide",
        "fidelity_patterns",
    ),
    (
        "B.Com(Hons)_IIIyearVIsem_FundamentalsofInvestments_Week2_DrKanuJain.pdf",
        "Fundamentals of Investments",
        "investments_fundamentals",
    ),
]

DB_PATH      = Path(__file__).parent / "rag_db"
REPORTS_DIR  = Path(__file__).parent / "reports"
CHUNK_WORDS  = 300   # target words per chunk
OVERLAP      = 60    # overlap words between consecutive chunks
MIN_CHUNK    = 60    # discard chunks shorter than this


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise extracted PDF text."""
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"-{2,}", "—", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_WORDS, overlap: int = OVERLAP):
    """Split text into overlapping word-windows."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) >= MIN_CHUNK:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def extract_symbol(pdf_path: Path) -> str:
    """
    Derive NSE symbol from file path.
      reports/TECHM.pdf              -> TECHM
      reports/RELIANCE_AR2024.pdf    -> RELIANCE
      reports/TCS/Q4FY25.pdf         -> TCS   (from subdirectory name)
    """
    # If file is inside a subdirectory of reports/, use the dir name
    if pdf_path.parent != REPORTS_DIR and pdf_path.parent.parent == REPORTS_DIR:
        return pdf_path.parent.name.upper()
    # Flat file — use stem, take first token before _ - or space
    stem = pdf_path.stem.upper()
    for sep in ("_", "-", " "):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


# ── Collection builders ───────────────────────────────────────────────────────

def build_trading_docs(client, ef):
    """(Re)build the trading_docs collection from textbook PDFs."""
    import pdfplumber

    try:
        client.delete_collection("trading_docs")
        print("  Deleted existing trading_docs collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name="trading_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for filename, source_name, source_id in PDFS:
        filepath = PDF_DIR / filename
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found — skipping")
            continue

        print(f"\n  Processing textbook: {filename}")
        fc, docs, metas, ids = 0, [], [], []

        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw  = page.extract_text() or ""
                text = clean_text(raw)
                if len(text) < MIN_CHUNK:
                    continue
                for chunk in chunk_text(text):
                    docs.append(chunk)
                    metas.append({
                        "source":    source_name,
                        "source_id": source_id,
                        "page":      page_num,
                        "file":      filename,
                        "type":      "textbook",
                    })
                    ids.append(f"{source_id}_p{page_num}_{fc}")
                    fc += 1

        for start in range(0, len(docs), 100):
            collection.add(
                documents=docs[start:start+100],
                metadatas=metas[start:start+100],
                ids=ids[start:start+100],
            )
        total += fc
        print(f"    {fc} chunks  (total so far: {total})")

    print(f"\n  trading_docs: {total} chunks")
    return total


def build_annual_reports(client, ef):
    """(Re)build the annual_reports collection from reports/ folder."""
    import pdfplumber

    if not REPORTS_DIR.exists():
        print(f"  reports/ folder not found at {REPORTS_DIR} — skipping")
        return 0

    # Gather all PDFs (flat + one level of subdirectories)
    pdf_files = sorted(REPORTS_DIR.rglob("*.pdf"))
    if not pdf_files:
        print("  No PDF files found in reports/ — skipping")
        return 0

    # Clean rebuild
    try:
        client.delete_collection("annual_reports")
        print("  Deleted existing annual_reports collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name="annual_reports",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    total      = 0
    seen_ids   = set()

    for pdf_path in pdf_files:
        symbol      = extract_symbol(pdf_path)
        source_name = f"{symbol} Annual Report"
        source_id   = f"ar_{symbol.lower()}"

        print(f"\n  Processing report: {pdf_path.name}  (symbol: {symbol})")
        fc, docs, metas, ids = 0, [], [], []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    raw  = page.extract_text() or ""
                    text = clean_text(raw)
                    if len(text) < MIN_CHUNK:
                        continue
                    for chunk in chunk_text(text):
                        uid = f"{source_id}_{pdf_path.stem}_p{page_num}_{fc}"
                        if uid in seen_ids:
                            uid += f"_x{fc}"   # dedup safety
                        seen_ids.add(uid)
                        docs.append(chunk)
                        metas.append({
                            "source":    source_name,
                            "source_id": source_id,
                            "page":      page_num,
                            "file":      pdf_path.name,
                            "symbol":    symbol,
                            "type":      "annual_report",
                        })
                        ids.append(uid)
                        fc += 1
        except Exception as e:
            print(f"    ERROR reading {pdf_path.name}: {e}")
            continue

        for start in range(0, len(docs), 100):
            collection.add(
                documents=docs[start:start+100],
                metadatas=metas[start:start+100],
                ids=ids[start:start+100],
            )
        total += fc
        print(f"    {fc} chunks  (total so far: {total})")

    # Summary: unique symbols
    symbols = sorted({extract_symbol(p) for p in pdf_files})
    print(f"\n  annual_reports: {total} chunks across {len(symbols)} companies: {', '.join(symbols)}")
    return total


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        sys.exit("pdfplumber not installed — run: pip install pdfplumber")
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        sys.exit("chromadb / sentence-transformers not installed")

    reports_only = "--reports-only" in sys.argv

    print("Initialising ChromaDB …")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    ef     = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
    )

    if reports_only:
        print("\n--- Annual reports rebuild only ---")
        n = build_annual_reports(client, ef)
        print(f"\nDone — {n} report chunks stored in {DB_PATH}")
    else:
        print("\n--- Full rebuild ---")
        n1 = build_trading_docs(client, ef)
        n2 = build_annual_reports(client, ef)
        print(f"\nDone — {n1} textbook + {n2} annual-report chunks stored in {DB_PATH}")

    print("Restart the server to pick up the changes.")


if __name__ == "__main__":
    main()
