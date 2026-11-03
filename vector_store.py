"""
Vector DB layer over the research paper PDF ONLY.
Structured data (MOESM3/5) deliberately stays OUT of here -
see structured_retrieval.py for why.
"""
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from app.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS,
    SIMILARITY_THRESHOLD,
    TOP_K_CHUNKS,
)


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


class PaperVectorStore:
    def __init__(self, pdf_path: str, rebuild: bool = False):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        # Default embedding function (sentence-transformers, downloads on first run)
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

        existing = [c.name for c in self.client.list_collections()]
        if rebuild and CHROMA_COLLECTION_NAME in existing:
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
            existing.remove(CHROMA_COLLECTION_NAME)

        if CHROMA_COLLECTION_NAME in existing:
            self.collection = self.client.get_collection(
                CHROMA_COLLECTION_NAME, embedding_function=self.embed_fn
            )
        else:
            self.collection = self.client.create_collection(
                CHROMA_COLLECTION_NAME, embedding_function=self.embed_fn
            )
            self._index_pdf(pdf_path)

    def _index_pdf(self, pdf_path: str):
        text = extract_pdf_text(pdf_path)
        chunks = chunk_text(text)
        self.collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )

    def retrieve(self, query: str, k: int = TOP_K_CHUNKS) -> list[str]:
        """
        Returns relevant chunks, or [] if nothing clears the similarity threshold.
        Chroma returns *distances* (lower = more similar) for the default embedding fn,
        so we convert to a similarity-ish score and filter.
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        kept = []
        for doc, dist in zip(docs, distances):
            similarity = 1 - dist  # rough conversion; tune SIMILARITY_THRESHOLD empirically
            if similarity >= SIMILARITY_THRESHOLD:
                kept.append(doc)
        return kept
