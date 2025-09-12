# main.py
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from together import Together

from backend.db import get_db_connection, create_legal_tables

# -------------------- Config --------------------
DEFAULT_LEGAL_MODEL = os.getenv("DEFAULT_LEGAL_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise RuntimeError("TOGETHER_API_KEY is not set in the environment.")

# -------------------- App --------------------
app = FastAPI(
    title="AI Legal Document Analyzer",
    description="Advanced legal document analysis with Together AI",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Services --------------------
together_client = Together(api_key=TOGETHER_API_KEY)

# Lazy-initialized on startup
legal_vector_store = None  # type: ignore

# -------------------- Models --------------------
class DocumentUploadResponse(BaseModel):
    success: bool
    doc_id: int
    filename: str
    document_type: Optional[str]
    chunks_created: int
    metadata: Dict[str, Any]
    message: str
    chunking_method: str = "semantic"

class RAGQuery(BaseModel):
    query: str
    top_k: int = 5
    document_type: Optional[str] = None
    practice_area: Optional[str] = None

class SourceItem(BaseModel):
    text: str
    source: str
    type: Optional[str] = None
    section: Optional[str] = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    legal_analysis: Dict[str, Any]
    confidence_score: float

# -------------------- Startup --------------------
@app.on_event("startup")
async def startup_event():
    """Initialize DB schema and create vector store service lazily."""
    try:
        create_legal_tables()
        print("✅ Legal database tables initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")

    # Import here to avoid early import issues
    global legal_vector_store
    from backend.vector_store import LegalVectorStore
    legal_vector_store = LegalVectorStore()
    print("✅ LegalVectorStore ready")

# -------------------- Health --------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Legal Document Analyzer", "version": "2.1.0"}

# -------------------- Upload & Store --------------------
@app.post("/upload-and-store", response_model=DocumentUploadResponse)
async def upload_and_store_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    practice_area: Optional[str] = Form(None),
):
    """Upload a legal document, parse → chunk → embed → store."""
    if legal_vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized yet")
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    try:
        file_bytes = await file.read()
        result = legal_vector_store.store_legal_document(
            filename=file.filename,
            file_bytes=file_bytes,
            document_type=document_type,
            practice_area=practice_area,
        )
        return DocumentUploadResponse(
            success=True,
            doc_id=result["doc_id"],
            filename=file.filename,
            document_type=result.get("document_type"),
            chunks_created=result["chunks_created"],
            metadata=result["metadata"],
            message=f"Successfully processed {file.filename} with {result['chunks_created']} chunks",
            chunking_method=result.get("chunking_method", "semantic"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

# -------------------- Agentic RAG Answer --------------------
@app.post("/agentic-rag-answer", response_model=RAGResponse)
async def agentic_rag_analysis(request: RAGQuery):
    """
    Hybrid search (vector + FTS + RRF + optional rerank) → build grounded context →
    LLM composes a clear, step-by-step legal answer.
    """
    if legal_vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized yet")

    try:
        # Use the unified hybrid_search method
        search_results = legal_vector_store.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            document_type=request.document_type,
            practice_area=request.practice_area,
            rerank=True,
        )
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        # Normalize sources for response + context building
        sources: List[SourceItem] = []
        for r in search_results:
            md = r.get("metadata", {})
            sources.append(
                SourceItem(
                    text=r.get("text", ""),
                    source=md.get("filename", "N/A"),
                    type=md.get("document_type"),
                    section=md.get("section_title", "N/A"),
                    rrf_score=float(r.get("rrf_score", 0.0)),
                    rerank_score=float(r.get("rerank_score", 0.0)),
                )
            )

        # Build compact context for the LLM
        legal_context = "\n\n".join(
            [f"Source: {s.source} / Section: {s.section}\nContent: {s.text}" for s in sources]
        )

        simple_prompt = (
            "Answer the legal question using only the information provided below. "
            "Be precise and explain step-by-step in simple language. "
            "If amounts, dates, timeframes, or procedures are present, include them. "
            "Use short sections or bullet points when helpful. "
            "If the answer is not in the sources, say what is missing.\n\n"
            f"Question: {request.query}\n\n"
            f"SOURCES:\n{legal_context}\n\n"
            "Answer:"
        )

        response = together_client.chat.completions.create(
            model=DEFAULT_LEGAL_MODEL,
            messages=[{"role": "user", "content": simple_prompt}],
            max_tokens=1000,
            temperature=0.1,
        )
        answer = response.choices[0].message.content

        # Confidence proxy
        rerank_scores = [s.rerank_score for s in sources if s.rerank_score > 0]
        if rerank_scores:
            confidence_score = float(sum(rerank_scores) / len(rerank_scores))
        else:
            rrf_scores = [s.rrf_score for s in sources]
            confidence_score = float(sum(rrf_scores) / len(rrf_scores)) if rrf_scores else 0.0

        legal_analysis = {
            "document_types_analyzed": sorted(set([s.type for s in sources if s.type])),
            "total_sources": len(sources),
            "avg_rrf_score": float(sum(s.rrf_score for s in sources) / len(sources)) if sources else 0.0,
            "avg_rerank_score": float(sum(s.rerank_score for s in sources) / len(sources)) if sources else 0.0,
            "search_method": "db_hybrid_reranked",
        }

        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            confidence_score=confidence_score,
            legal_analysis=legal_analysis,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic RAG analysis failed: {e}")

# -------------------- Summarize a Document --------------------
@app.post("/document-summary")
async def document_summary(
    doc_id: int = Query(..., description="ID from legal_documents.doc_id"),
    summary_type: str = Query("comprehensive", pattern="^(executive|comprehensive)$"),
):
    """Generate an executive or comprehensive summary for a stored document."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT filename, full_text, document_type FROM legal_documents WHERE doc_id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found.")

        filename, full_text, doc_type = row
        prompt = (
            f"Provide a {summary_type} summary of this legal document "
            f"'{filename}' ({doc_type}):\n\n{(full_text or '')[:8000]}"
        )

        response = together_client.chat.completions.create(
            model=DEFAULT_LEGAL_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert legal analyst specializing in document summarization."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        return {
            "document_id": doc_id,
            "filename": filename,
            "summary_type": summary_type,
            "summary": response.choices[0].message.content,
            "generated_at": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document summary failed: {e}")

# -------------------- Hybrid Search (JSON) --------------------
@app.get("/search-documents")
async def search_documents(
    query: str = Query(..., description="Your search query"),
    limit: int = Query(10, ge=1, le=50),
    document_type: Optional[str] = Query(None),
    practice_area: Optional[str] = Query(None),
    rerank: bool = Query(False, description="Set true to enable cross-encoder reranking"),
):
    """Hybrid search: vector + FTS + RRF (+ optional rerank)."""
    if legal_vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized yet")

    try:
        results = legal_vector_store.hybrid_search(
            query=query,
            top_k=limit,
            document_type=document_type,
            practice_area=practice_area,
            rerank=rerank,
        )

        formatted = [
            {
                "filename": r.get("metadata", {}).get("filename", "N/A"),
                "document_type": r.get("metadata", {}).get("document_type"),
                "practice_area": r.get("metadata", {}).get("practice_area"),
                "section_title": r.get("metadata", {}).get("section_title"),
                "chunk_id": r.get("metadata", {}).get("chunk_id"),
                "rrf_score": r.get("rrf_score", 0.0),
                "rerank_score": r.get("rerank_score", 0.0),
                "chunk_text": r.get("text", ""),
            }
            for r in results
        ]

        return {"total_results": len(formatted), "results": formatted}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document search failed: {e}")

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
