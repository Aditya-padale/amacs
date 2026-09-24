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
