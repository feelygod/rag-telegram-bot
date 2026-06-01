import hashlib
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "index.json"
        self.docs_file = self.storage_dir / "docs.json"
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,
            ngram_range=(1, 2),
        )
        self._docs: list[dict] = []
        self._vectors: np.ndarray = np.empty((0, 5000))
        self._load()

    def _load(self):
        if self.docs_file.exists():
            self._docs = json.loads(self.docs_file.read_text(encoding="utf-8"))
        if self._docs:
            self._rebuild_index()

    def _save(self):
        self.docs_file.write_text(json.dumps(self._docs, ensure_ascii=False), encoding="utf-8")

    def _rebuild_index(self):
        texts = [d["text"] for d in self._docs]
        if len(texts) == 1:
            texts = texts + [""]
        try:
            self._vectors = self.vectorizer.fit_transform(texts).toarray()
            if len(self._docs) == 1:
                self._vectors = self._vectors[:1]
        except ValueError:
            self._vectors = np.empty((0, 5000))

    def add_document(self, text: str, metadata: Optional[dict] = None) -> str:
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        for doc in self._docs:
            if doc["id"] == doc_id:
                return doc_id
        chunks = self._chunk_text(text)
        for chunk in chunks:
            self._docs.append({
                "id": hashlib.md5(chunk.encode()).hexdigest()[:12],
                "text": chunk,
                "metadata": metadata or {},
            })
        self._rebuild_index()
        self._save()
        return doc_id

    def search(self, query: str, k: int = 5) -> List[dict]:
        if not self._docs or self._vectors.shape[0] == 0:
            return []
        try:
            query_vec = self.vectorizer.transform([query]).toarray()
            scores = cosine_similarity(query_vec, self._vectors)[0]
            top_idx = np.argsort(scores)[::-1][:k]
            results = []
            for i in top_idx:
                results.append({
                    "id": self._docs[i]["id"],
                    "text": self._docs[i]["text"],
                    "score": float(scores[i]),
                    "metadata": self._docs[i]["metadata"],
                })
            return results
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
        self._docs = []
        self._vectors = np.empty((0, 5000))
        self._save()
        if self.index_file.exists():
            self.index_file.unlink()

    def count_documents(self) -> int:
        return len(self._docs)
