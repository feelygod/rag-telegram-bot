import hashlib
import os
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _get_db():
    """Returns a DB connection. Uses PostgreSQL (via DATABASE_URL) or falls back to SQLite."""
    url = os.getenv("DATABASE_URL", "")
    if url:
        import psycopg2
        conn = psycopg2.connect(url)
        return conn
    import sqlite3
    storage_dir = Path(os.getenv("STORAGE_DIR", "storage"))
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = storage_dir / "docs.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn):
    url = os.getenv("DATABASE_URL", "")
    if url:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                filename TEXT DEFAULT ''
            )
        """)
        conn.commit()
        cur.close()
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                filename TEXT DEFAULT ''
            )
        """)
        conn.commit()


def _fetch_all(conn):
    url = os.getenv("DATABASE_URL", "")
    if url:
        cur = conn.cursor(cursor_factory=__import__('psycopg2').extras.RealDictCursor)
        cur.execute("SELECT id, text, filename FROM chunks ORDER BY id")
        rows = cur.fetchall()
        cur.close()
        return rows
    rows = conn.execute("SELECT id, text, filename FROM chunks ORDER BY id").fetchall()
    return rows


def _insert_chunk(conn, cid, chunk, filename):
    url = os.getenv("DATABASE_URL", "")
    if url:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chunks (id, text, filename) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (cid, chunk, filename),
        )
        conn.commit()
        cur.close()
    else:
        conn.execute(
            "INSERT OR IGNORE INTO chunks (id, text, filename) VALUES (?, ?, ?)",
            (cid, chunk, filename),
        )
        conn.commit()


def _delete_all(conn):
    url = os.getenv("DATABASE_URL", "")
    if url:
        cur = conn.cursor()
        cur.execute("DELETE FROM chunks")
        conn.commit()
        cur.close()
    else:
        conn.execute("DELETE FROM chunks")
        conn.commit()


class RAGEngine:
    def __init__(self, storage_dir: str = "storage"):
        os.environ.setdefault("STORAGE_DIR", storage_dir)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,
            ngram_range=(1, 2),
        )
        self._docs: list[dict] = []
        self._vectors: np.ndarray = np.empty((0, 5000))
        conn = _get_db()
        _init_db(conn)
        conn.close()
        self._load_from_db()

    def _load_from_db(self):
        conn = _get_db()
        rows = _fetch_all(conn)
        conn.close()
        self._docs = [
            {"id": r["id"], "text": r["text"], "metadata": {"filename": r["filename"]}}
            for r in rows
        ]
        self._rebuild_index()

    def _rebuild_index(self):
        if len(self._docs) == 0:
            self._vectors = np.empty((0, 5000))
            return
        texts = [d["text"] for d in self._docs]
        if len(texts) == 1:
            texts = texts + [""]
        try:
            vecs = self.vectorizer.fit_transform(texts).toarray()
            self._vectors = vecs[:1] if len(self._docs) == 1 else vecs
        except ValueError:
            self._vectors = np.empty((0, 5000))

    def add_document(self, text: str, metadata: Optional[dict] = None) -> str:
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        for doc in self._docs:
            if doc["id"] == doc_id:
                return doc_id
        chunks = self._chunk_text(text)
        filename = (metadata or {}).get("filename", "")
        conn = _get_db()
        for chunk in chunks:
            cid = hashlib.md5(chunk.encode()).hexdigest()[:12]
            self._docs.append({
                "id": cid,
                "text": chunk,
                "metadata": {"filename": filename},
            })
            _insert_chunk(conn, cid, chunk, filename)
        conn.close()
        self._rebuild_index()
        return doc_id

    def search(self, query: str, k: int = 5) -> List[dict]:
        if not self._docs or self._vectors.shape[0] == 0:
            return []
        try:
            query_vec = self.vectorizer.transform([query]).toarray()
            scores = cosine_similarity(query_vec, self._vectors)[0]
            top_idx = np.argsort(scores)[::-1][:k]
            return [
                {
                    "id": self._docs[i]["id"],
                    "text": self._docs[i]["text"],
                    "score": float(scores[i]),
                    "metadata": self._docs[i]["metadata"],
                }
                for i in top_idx
            ]
        except ValueError:
            return []

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks if chunks else [text]

    def clear(self):
        conn = _get_db()
        _delete_all(conn)
        conn.close()
        self._docs = []
        self._vectors = np.empty((0, 5000))

    def count_documents(self) -> int:
        return len(self._docs)
