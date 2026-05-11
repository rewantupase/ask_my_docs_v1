"""
src/ingestion.py — Document loading and chunking.

Supports PDF, Markdown, and web pages.
Every chunk gets metadata injected so citations can reference the exact source,
page number, and character offset.

Why RecursiveCharacterTextSplitter?
  It tries to split on natural language boundaries first:
    paragraph break → line break → sentence end → word boundary → character
  This keeps semantically coherent ideas together in one chunk, rather than
  cutting mid-sentence at a hard character limit.

Why 600 tokens / 100 overlap?
  - Too small (< 200 tokens): chunks lack enough context to answer questions.
  - Too large (> 1200 tokens): retrieval becomes imprecise; too much noise
    surrounds the relevant sentence.
  - 100-token overlap: ensures sentences near chunk boundaries appear in both
    adjacent chunks, preventing lost context at the seam.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Union

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pdf(path: str) -> List[Document]:
    """
    Load a PDF using PyMuPDF (fitz).
    PyMuPDF preserves reading order and handles two-column layouts (common
    in research papers) far better than PyPDF2 or pdfminer.
    """
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
    except ImportError:
        raise ImportError("pip install pymupdf")

    loader = PyMuPDFLoader(path)
    docs = loader.load()
    logger.info(f"PDF loaded: {path} → {len(docs)} pages")
    return docs


def load_markdown(path: str) -> List[Document]:
    """Load a Markdown file, preserving header structure as metadata."""
    try:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
    except ImportError:
        raise ImportError("pip install unstructured")

    loader = UnstructuredMarkdownLoader(path, mode="elements")
    docs = loader.load()
    logger.info(f"Markdown loaded: {path} → {len(docs)} elements")
    return docs


def load_web(url: str) -> List[Document]:
    """
    Load a web page. Strips nav/footer/ads via BeautifulSoup and returns
    the main article text. Good for arXiv abstracts, blog posts, docs pages.
    """
    try:
        from langchain_community.document_loaders import WebBaseLoader
        import bs4
    except ImportError:
        raise ImportError("pip install beautifulsoup4")

    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={"parse_only": bs4.SoupStrainer(
            # Keep only main article content; skip nav, ads, footers
            ["article", "main", "div.content", "div.article", "p", "h1", "h2", "h3"]
        )},
    )
    docs = loader.load()
    logger.info(f"Web page loaded: {url} → {len(docs)} doc(s)")
    return docs


def load_source(source: str) -> List[Document]:
    """
    Auto-detect source type and dispatch to the right loader.
    source can be:
      - a file path ending in .pdf
      - a file path ending in .md or .markdown
      - an http/https URL
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_web(source)
    
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(str(path))
    elif suffix in (".md", ".markdown"):
        return load_markdown(str(path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF, Markdown, or a URL.")


def load_folder(folder: str) -> List[Document]:
    """Load all supported documents from a folder recursively."""
    folder_path = Path(folder)
    all_docs = []
    supported = {".pdf", ".md", ".markdown"}
    
    files = [f for f in folder_path.rglob("*") if f.suffix.lower() in supported]
    logger.info(f"Found {len(files)} document(s) in {folder}")
    
    for f in files:
        try:
            docs = load_source(str(f))
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")
    
    return all_docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks and inject rich metadata.

    The splitter tries separators in order:
      1. \\n\\n  — paragraph breaks (best: keeps paragraphs together)
      2. \\n    — line breaks
      3. ". "   — sentence ends
      4. " "    — word boundaries
      5. ""     — character-level (last resort)

    Metadata injected per chunk:
      - chunk_id        : globally unique identifier (hash of content)
      - chunk_index     : sequential index for citation numbering
      - source          : original file path or URL
      - page            : page number (PDFs) or 0 (others)
      - chunk_text_prev : preview of content for citation display
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=config.CHUNK_SEPARATORS,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} document(s)")

    for i, chunk in enumerate(chunks):
        # Stable unique ID: hash of content so re-ingesting the same doc
        # doesn't create duplicate vectors in ChromaDB
        content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
        
        chunk.metadata["chunk_id"]    = f"chunk_{content_hash}"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["page"]        = chunk.metadata.get("page", 0)
        chunk.metadata["source"]      = chunk.metadata.get("source", "unknown")
        # 300-char preview shown in citation output
        chunk.metadata["preview"]     = chunk.page_content[:300].replace("\n", " ")

    return chunks


# ── Convenience ───────────────────────────────────────────────────────────────

def ingest(source: Union[str, List[str]]) -> List[Document]:
    """
    Full ingestion pipeline: load → chunk.
    Accepts a single source string or a list of sources.
    Returns a flat list of chunk Documents ready for embedding.
    """
    if isinstance(source, list):
        all_docs = []
        for s in source:
            all_docs.extend(load_source(s))
        raw_docs = all_docs
    else:
        raw_docs = load_source(source)

    return chunk_documents(raw_docs)
