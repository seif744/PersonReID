"""
store.py  --  STAGE 6 (storage):  Qdrant vector store for ReID embeddings.

============================ WHAT THIS IS =================================
A thin, production-minded wrapper around Qdrant whose ONLY job is:
    * store person embeddings (512-d vectors) + their metadata, and
    * given a new embedding, return the nearest stored ones (candidates).

It is a STORAGE + RETRIEVAL layer. It deliberately does NOT decide identity
(see src/identity/DESIGN.md): it returns candidates with similarity scores; a
smarter service turns candidates into a global_id. Keeping that line is what
stops this project from turning into spaghetti.

WHY a wrapper instead of calling qdrant_client everywhere:
  * one place owns the collection name, vector size, and distance metric, so
    they can't drift out of sync across the codebase;
  * one place validates inputs (vector size) and fails with a clear message;
  * callers speak in numpy embeddings + plain dicts, not Qdrant's PointStruct.

------------------------------- KEY CHOICES -------------------------------
* Vector size = 512, metric = COSINE. These MUST match ReIDExtractor's output
  (512-d, L2-normalized). Cosine is the "0.82 same / 0.55 different" score you
  already saw. If the backbone ever changes size, change EMBEDDING_DIM here and
  recreate the collection -- old vectors of a different size are incompatible.

* Point ids are UUIDs. Each stored embedding is one OBSERVATION of a person at a
  moment; UUIDs guarantee we never collide across cameras, restarts, or workers.
  (The person's cross-camera identity is a PAYLOAD field, not the point id.)

* Persistent local storage by default (path=...). Note: local mode locks the
  folder to ONE process -- fine for dev. For real deployment (many camera
  workers writing at once) you pass a client pointed at a Qdrant SERVER instead;
  the rest of this class is unchanged.
============================================================================
"""

import uuid
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter

# Must match ReIDExtractor.EMBEDDING_DIM. Duplicated as a constant here (not
# imported from reid) on purpose: the storage layer must not depend on the model
# layer. If they ever disagree, the dimension guard below fails loudly.
EMBEDDING_DIM = 512
DEFAULT_COLLECTION = "persons"


class PersonVectorStore:
    def __init__(self, path: str = "qdrant_data", *,
                 client: Optional[QdrantClient] = None,
                 collection: str = DEFAULT_COLLECTION,
                 dim: int = EMBEDDING_DIM):
        """
        path       : folder for persistent local storage. Use ":memory:" for a
                     throwaway in-process store (tests).
        client     : pass your own QdrantClient (e.g. pointed at a server) to
                     override `path`. This is the deployment path.
        collection : the collection ("table") name.
        dim        : embedding size; guards every insert/search.
        """
        self.client = client or QdrantClient(path=path)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Create the collection only if it doesn't already exist. Idempotent: safe
        to construct the store on every process start / restart without wiping or
        crashing on existing data.
        """
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def add(self, embedding: np.ndarray, payload: dict) -> str:
        """Store ONE embedding + its metadata. Returns the new point's id."""
        return self.add_many([embedding], [payload])[0]

    def add_many(self, embeddings, payloads: List[dict]) -> List[str]:
        """
        Store many embeddings in ONE call (a single upsert is far cheaper than N).
        `embeddings` and `payloads` must line up 1:1.
        """
        if len(embeddings) != len(payloads):
            raise ValueError("embeddings and payloads must have the same length")

        ids = [uuid.uuid4().hex for _ in embeddings]
        points = [
            PointStruct(id=i, vector=self._as_vector(e), payload=p)
            for i, e, p in zip(ids, embeddings, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return ids

    def search(self, embedding: np.ndarray, limit: int = 5,
               query_filter: Optional[Filter] = None):
        """
        Return the `limit` nearest stored points to `embedding`, most-similar
        first. Each hit has .id, .score (cosine, higher=closer), and .payload.

        query_filter is an optional Qdrant Filter (e.g. restrict to a camera or a
        time window). The store just passes it through -- WHICH filter to apply is
        the Identity Service's policy decision, not the store's.
        """
        return self.client.query_points(
            collection_name=self.collection,
            query=self._as_vector(embedding),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points

    def count(self) -> int:
        """How many points are currently stored."""
        return self.client.count(self.collection).count

    def reset(self):
        """
        Drop and recreate the collection (wipe all points). For TESTS/DEMOS so
        reruns start clean. Never call this in production -- it deletes data.
        """
        self.client.delete_collection(self.collection)
        self._ensure_collection()

    def _as_vector(self, embedding) -> List[float]:
        """
        numpy embedding -> plain list[float] for Qdrant, with a size guard.
        A wrong-sized vector here means a mismatch between the model and the
        collection -- fail loud rather than store a corrupt point.
        """
        arr = np.asarray(embedding, dtype=np.float32).ravel()
        if arr.shape[0] != self.dim:
            raise ValueError(
                f"Embedding has size {arr.shape[0]}, expected {self.dim}. "
                f"Model/collection dimension mismatch."
            )
        return arr.tolist()
