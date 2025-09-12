# embeddings.py
import os
import re
from typing import List, Optional
from together import Together


class LegalEmbeddings:
    """
    Turn text into embedding vectors using the Together API.

    - Cleans & truncates input (default: 400 chars)
    - Embeds in small batches (default: 10)
    - Falls back to per-item on batch failure
    - Returns zero-vector on any final failure (keeps order & length)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_chars: int = 400,
        batch_size: int = 10,
        embedding_dim: int = 768,
    ):
        self.client = Together(api_key=api_key or os.getenv("TOGETHER_API_KEY"))
        self.model = model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.max_chars = max_chars
        self.batch_size = batch_size
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", embedding_dim))
        print(f"Initialized embeddings with model: {self.model}")

    # ----------------- Public API -----------------

    def get_embedding(self, text: str) -> List[float]:
        """Embed a single string."""
        cleaned = self._clean(text)
        try:
            resp = self.client.embeddings.create(input=[cleaned], model=self.model)
            return resp.data[0].embedding
        except Exception as e:
            print(f"❌ Single embedding failed: {e}")
            return self._zero_vec()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed many strings (order preserved)."""
        if not texts:
            return []

        cleaned = [self._clean(t) for t in texts]
        print("=== BATCH EMBEDDING ===")
        print(f"Processing {len(cleaned)} texts (each ≤ {self.max_chars} chars)")

        all_embeds: List[List[float]] = []

        # Embed in batches
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            try:
                resp = self.client.embeddings.create(input=batch, model=self.model)
                all_embeds.extend([item.embedding for item in resp.data])
                done = start // self.batch_size + 1
                total = (len(cleaned) - 1) // self.batch_size + 1
                print(f"✅ Processed batch {done}/{total}")
            except Exception as e:
                print(f"❌ Batch failed (items {start}:{start+len(batch)}): {e}")
                # Fallback: try each item in this batch
                for idx, txt in enumerate(batch, start=start):
                    try:
                        r = self.client.embeddings.create(input=[txt], model=self.model)
                        all_embeds.append(r.data[0].embedding)
                    except Exception as e2:
                        print(f"   ❌ Item {idx} failed: {e2}")
                        all_embeds.append(self._zero_vec())

        print(f"✅ Completed {len(all_embeds)} embeddings")
        return all_embeds

    # ----------------- Helpers -----------------

    def _clean(self, text: str) -> str:
        """Trim, collapse spaces, and hard-truncate."""
        if not text or not text.strip():
            return "empty placeholder"
        text = text[: self.max_chars]
        text = re.sub(r"\s+", " ", text).strip()
        return text[: self.max_chars]

    def _zero_vec(self) -> List[float]:
        """Return a zero vector (fallback) with correct dimension."""
        return [0.0] * self.embedding_dim
