"""Vector DB adapter stubs — enabled via ``pip install amacs[pinecone]`` or ``amacs[faiss]``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from amacs.exceptions import AMACSError


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Insert or update vectors."""

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query for nearest neighbours."""


class SimpleVectorStore(VectorStore):
    """In-memory vector and document store adapter for keyword/semantic matching."""

    def __init__(self, documents: Optional[List[str]] = None) -> None:
        self.documents: List[str] = documents or []

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        for v in vectors:
            content = v.get("content") or v.get("text") or str(v.get("metadata", {}))
            if content:
                self.documents.append(content)

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {"id": i, "score": 1.0, "metadata": {"text": doc}}
            for i, doc in enumerate(self.documents[:top_k])
        ]

    def search_text(self, query_text: str, top_k: int = 5) -> List[str]:
        words = set(query_text.lower().split())
        scored: List[tuple[int, str]] = []
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            overlap = len(words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class PineconeStore(VectorStore):
    """Pinecone adapter — requires ``pinecone-client``."""

    def __init__(self, index_name: str, **kwargs: Any) -> None:
        try:
            from pinecone import Pinecone  # type: ignore[import-untyped]
        except ImportError:
            raise AMACSError(
                "pinecone-client not installed. Run: pip install amacs[pinecone]"
            ) from None
        pc = Pinecone(**kwargs)
        self._index = pc.Index(index_name)

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        self._index.upsert(vectors=vectors)

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        resp = self._index.query(vector=vector, top_k=top_k, filter=filter)
        return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in resp.matches]


class FAISSStore(VectorStore):
    """FAISS adapter — requires ``faiss-cpu``."""

    def __init__(self, dimension: int) -> None:
        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError:
            raise AMACSError(
                "faiss-cpu not installed. Run: pip install amacs[faiss]"
            ) from None
        self._index = faiss.IndexFlatL2(dimension)
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        import numpy as np  # type: ignore[import-untyped]

        for v in vectors:
            vec = np.array([v["values"]], dtype="float32")
            self._index.add(vec)
            self._metadata[self._next_id] = v.get("metadata", {})
            self._next_id += 1

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        import numpy as np  # type: ignore[import-untyped]

        vec = np.array([vector], dtype="float32")
        distances, indices = self._index.search(vec, top_k)
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {"id": int(idx), "score": float(dist), "metadata": self._metadata.get(int(idx), {})}
            )
        return results
