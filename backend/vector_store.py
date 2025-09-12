# backend/vector_store.py
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from backend.db import get_db_connection
from backend.embeddings import LegalEmbeddings
from backend.semantic_chunker import SemanticChunker
from backend.file_parser import LegalDocumentParser


class LegalVectorStore:
    def __init__(self):
        self.parser = LegalDocumentParser()
        self.chunker = SemanticChunker()
        self.emb = LegalEmbeddings()

    # ---------------- Ingest & Store ----------------
    def store_legal_document(
        self,
        filename: str,
        file_bytes: bytes,
        document_type: Optional[str] = None,
        practice_area: Optional[str] = None,
    ) -> Dict:
        parsed = self.parser.extract_text(filename, file_bytes)
        text, meta = parsed["text"], parsed["metadata"]

        chunks = self.chunker.semantic_chunk_legal_text(text)
        if not chunks:
            raise ValueError("No text chunks generated")

        chunk_texts = [c["text"] for c in chunks]
        chunk_vecs = self.emb.get_embeddings_batch(chunk_texts)
        doc_vec = np.mean(chunk_vecs, axis=0).tolist()

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO legal_documents (filename, document_type, practice_area, full_text, embedding, metadata)
                VALUES (%s,%s,%s,%s,%s,%s)
                RETURNING doc_id
                """,
                (
                    filename[:500],
                    (document_type or "")[:200],
                    (practice_area or "")[:200],
                    text,
                    doc_vec,
                    json.dumps(meta),
                ),
            )
            doc_id = cur.fetchone()[0]

            rows = []
            for i, ch in enumerate(chunks):
                if i >= len(chunk_vecs):
                    break
                rows.append(
                    (
                        doc_id,
                        ch["text"],
                        (ch.get("type", "semantic_chunk"))[:200],
                        (ch.get("section_title", ""))[:500],
                        chunk_vecs[i],
                        ch.get("sentence_count", 1),
                        json.dumps(
                            {
                                "avg_similarity": ch.get("avg_similarity"),
                                "original_type": ch.get("type"),
                            }
                        ),
                    )
                )

            if rows:
                cur.executemany(
                    """
                    INSERT INTO legal_chunks
                    (doc_id, chunk_text, chunk_type, section_title, embedding, sentence_count, chunk_metadata)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    rows,
                )

            conn.commit()
            return {
                "doc_id": doc_id,
                "chunks_created": len(rows),
                "document_type": document_type,
                "metadata": meta,
                "chunking_method": "semantic",
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    # ---------------- Hybrid Search (DB) ----------------
    def _rrf(self, *rank_lists: List[Tuple[int, float]], k: int = 60) -> List[Tuple[int, float]]:
        scores = defaultdict(float)
        for rlist in rank_lists:
            for rank, (cid, _score) in enumerate(rlist, 1):
                scores[cid] += 1.0 / (rank + k)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        document_type: Optional[str] = None,
        practice_area: Optional[str] = None,
        rerank: bool = True,
    ) -> List[Dict]:
        qvec = self.emb.get_embedding(query)

        conn = get_db_connection()
        cur = conn.cursor()

        # 1) dense (vector) search
        v_sql = """
            SELECT c.chunk_id, (c.embedding <=> %s::vector) AS distance
            FROM legal_chunks c JOIN legal_documents d ON c.doc_id = d.doc_id
            WHERE 1=1
        """
        v_args = [qvec]
        if document_type:
            v_sql += " AND d.document_type = %s"
            v_args.append(document_type)
        if practice_area:
            v_sql += " AND d.practice_area = %s"
            v_args.append(practice_area)
        v_sql += " ORDER BY distance ASC LIMIT 50"
        cur.execute(v_sql, v_args)
        vector_hits = [(cid, dist) for cid, dist in cur.fetchall()]  # smaller=better

        # 2) sparse (full-text) search
        f_sql = """
            SELECT c.chunk_id, ts_rank(c.tsv, plainto_tsquery('english', %s)) AS rank
            FROM legal_chunks c JOIN legal_documents d ON c.doc_id = d.doc_id
            WHERE c.tsv @@ plainto_tsquery('english', %s)
        """
        f_args = [query, query]
        if document_type:
            f_sql += " AND d.document_type = %s"
            f_args.append(document_type)
        if practice_area:
            f_sql += " AND d.practice_area = %s"
            f_args.append(practice_area)
        f_sql += " ORDER BY rank DESC LIMIT 50"
        cur.execute(f_sql, f_args)
        fts_hits = [(cid, rank) for cid, rank in cur.fetchall()]  # larger=better

        # 3) fuse (RRF)
        fused = self._rrf(vector_hits, fts_hits)
        if not fused:
            cur.close()
            conn.close()
            return []

        # 4) fetch top text/meta
        top_ids = [cid for cid, _ in fused[: top_k * 2]]
        cur.execute(
            """
            SELECT c.chunk_id, c.chunk_text, d.filename, d.document_type, d.practice_area, c.section_title
            FROM legal_chunks c JOIN legal_documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id = ANY(%s)
            """,
            (top_ids,),
        )
        rowmap = {
            cid: (text, fn, dt, pa, sec) for cid, text, fn, dt, pa, sec in cur.fetchall()
        }
        cur.close()
        conn.close()

        initial = []
        for cid, rrf_score in fused:
            if cid in rowmap:
                text, fn, dt, pa, sec = rowmap[cid]
                initial.append(
                    {
                        "text": text,
                        "metadata": {
                            "chunk_id": cid,
                            "filename": fn,
                            "document_type": dt,
                            "practice_area": pa,
                            "section_title": sec,
                        },
                        "rrf_score": rrf_score,
                    }
                )

        # 5) optional rerank (Together cross-encoder)
        if rerank and initial:
            try:
                from together import Together
                client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
                docs = [r["text"] for r in initial]
                resp = client.rerank.create(
                    model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"),
                    query=query,
                    documents=docs,
                    top_n=min(top_k, len(docs)),
                )
                out = []
                for item in resp.results:
                    r = initial[item.index].copy()
                    r["rerank_score"] = item.relevance_score
                    out.append(r)
                return out[:top_k]
            except Exception:
                # If rerank fails, return RRF results
                pass

        return initial[:top_k]

    # Backward-compat (if any old code calls the old name)
    def hybrid_search_legal_documents(
        self,
        query: str,
        top_k: int = 10,
        document_type: Optional[str] = None,
        practice_area: Optional[str] = None,
        use_reranking: bool = True,
    ) -> List[Dict]:
        return self.hybrid_search(
            query=query,
            top_k=top_k,
            document_type=document_type,
            practice_area=practice_area,
            rerank=use_reranking,
        )

    # Optional: vector-only search if referenced elsewhere
    def search_legal_documents(
        self,
        query: str,
        top_k: int = 5,
        document_type: Optional[str] = None,
        practice_area: Optional[str] = None,
    ) -> List[Dict]:
        qvec = self.emb.get_embedding(query)
        conn = get_db_connection()
        cur = conn.cursor()

        sql = """
            SELECT 
                c.chunk_id, c.doc_id, c.chunk_text, c.chunk_type, c.section_title,
                d.filename, d.document_type, d.practice_area,
                (c.embedding <=> %s::vector) AS distance
            FROM legal_chunks c
            JOIN legal_documents d ON c.doc_id = d.doc_id
            WHERE 1=1
        """
        args = [qvec]
        if document_type:
            sql += " AND d.document_type = %s"
            args.append(document_type)
        if practice_area:
            sql += " AND d.practice_area = %s"
            args.append(practice_area)
        sql += " ORDER BY distance ASC LIMIT %s"
        args.append(top_k)

        cur.execute(sql, args)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            {
                "chunk_id": r[0],
                "doc_id": r[1],
                "chunk_text": r[2],
                "chunk_type": r[3],
                "section_title": r[4],
                "filename": r[5],
                "document_type": r[6],
                "practice_area": r[7],
                "similarity_score": 1 - float(r[8]),
            }
            for r in rows
        ]
