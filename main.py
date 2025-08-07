# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from together import Together
import json
from datetime import datetime

# Import your enhanced modules
from backend.db import get_db_connection, create_legal_tables
from backend.file_parser import LegalDocumentParser
from backend.vector_store import LegalVectorStore
from backend.embeddings import LegalEmbeddings

# Initialize FastAPI app
app = FastAPI(
    title="AI Legal Document Analyzer",
    description="Advanced legal document analysis with Together AI",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Together AI client
together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Initialize components
legal_parser = LegalDocumentParser()
legal_vector_store = LegalVectorStore()
legal_embeddings = LegalEmbeddings()

# Pydantic models for request/response
class DocumentUploadResponse(BaseModel):
    success: bool
    doc_id: int
    filename: str
    document_type: Optional[str]
    chunks_created: int
    metadata: Dict[str, Any]
    message: str

class RAGQuery(BaseModel):
    query: str
    top_k: int = 5
    document_type: Optional[str] = None
    practice_area: Optional[str] = None
    jurisdiction: Optional[str] = None

class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    legal_analysis: Dict[str, Any]
    confidence_score: float

class LegalAnalysisRequest(BaseModel):
    query: str
    document_types: Optional[List[str]] = None
    analysis_type: str = "general"  # general, contract, compliance, risk

class ContractAnalysisRequest(BaseModel):
    query: str
    contract_type: Optional[str] = None
    focus_areas: Optional[List[str]] = None  # liability, termination, payment, etc.

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        create_legal_tables()
        print("✅ Legal database tables initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Legal Document Analyzer"}

# Document upload and processing endpoint
@app.post("/upload-and-store", response_model=DocumentUploadResponse)
async def upload_and_store_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    jurisdiction: Optional[str] = Form(None),
    practice_area: Optional[str] = Form(None)
):
    """Upload and process legal document with enhanced metadata"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Store document with legal metadata
        result = legal_vector_store.store_legal_document(
            filename=file.filename,
            file_bytes=file_content,
            document_type=document_type,
            jurisdiction=jurisdiction,
            practice_area=practice_area
        )
        
        return DocumentUploadResponse(
            success=True,
            doc_id=result['doc_id'],
            filename=file.filename,
            document_type=result.get('document_type'),
            chunks_created=result['chunks_created'],
            metadata=result['metadata'],
            message=f"Successfully processed {file.filename} with {result['chunks_created']} chunks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

# Enhanced RAG endpoint with legal analysis
@app.post("/rag-answer", response_model=RAGResponse)
async def rag_legal_analysis(request: RAGQuery):
    """Advanced RAG with legal-specific analysis"""
    
    try:
        # Search for relevant legal documents
        search_results = legal_vector_store.search_legal_documents(
            query=request.query,
            top_k=request.top_k,
            document_type=request.document_type,
            practice_area=request.practice_area
        )
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Prepare context for legal analysis
        context_chunks = []
        for result in search_results:
            context_chunks.append({
                'text': result['chunk_text'],
                'source': result['filename'],
                'type': result['document_type'],
                'section': result.get('section_title', 'N/A'),
                'score': result['similarity_score']
            })
        
        # Create legal analysis prompt
        legal_context = "\n\n".join([
            f"Document: {chunk['source']} (Type: {chunk['type']})\n"
            f"Section: {chunk['section']}\n"
            f"Content: {chunk['text']}\n"
            f"Relevance Score: {chunk['score']:.3f}"
            for chunk in context_chunks
        ])
        
        legal_prompt = f"""
        You are an expert legal AI assistant specializing in document analysis. 
        Analyze the following legal documents and provide a comprehensive answer to the user's question.
        
        User Question: {request.query}
        
        Legal Document Context:
        {legal_context}
        
        Please provide:
        1. A direct answer to the question
        2. Legal analysis and implications
        3. Key legal concepts identified
        4. Potential risks or considerations
        5. Relevant legal citations if present
        
        Format your response as a comprehensive legal analysis while being clear and accessible.
        """
        
        # Get legal analysis from Together AI
        response = together_client.chat.completions.create(
            model=os.getenv("DEFAULT_LEGAL_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
            messages=[
                {"role": "system", "content": "You are an expert legal AI assistant."},
                {"role": "user", "content": legal_prompt}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        # Calculate confidence score based on similarity scores
        avg_similarity = sum(r['similarity_score'] for r in search_results) / len(search_results)
        confidence_score = min(0.95, max(0.1, 1.0 - avg_similarity))
        
        # Extract legal analysis components
        legal_analysis = {
            "document_types_analyzed": list(set(r['document_type'] for r in search_results if r['document_type'])),
            "jurisdictions": list(set(r['jurisdiction'] for r in search_results if r['jurisdiction'])),
            "practice_areas": list(set(r['practice_area'] for r in search_results if r['practice_area'])),
            "total_sources": len(search_results),
            "avg_relevance": avg_similarity
        }
        
        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=context_chunks,
            legal_analysis=legal_analysis,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legal analysis failed: {str(e)}")

# Contract-specific analysis endpoint
@app.post("/contract-analysis")
async def contract_analysis(request: ContractAnalysisRequest):
    """Specialized contract analysis endpoint"""
    
    try:
        # Search for contract documents
        search_results = legal_vector_store.search_legal_documents(
            query=request.query,
            top_k=10,
            document_type="contract"
        )
        
        if not search_results:
            return {"error": "No contract documents found"}
        
        # Prepare contract-specific context
        contract_context = "\n\n".join([
            f"Contract: {result['filename']}\n"
            f"Section: {result.get('section_title', 'General')}\n"
            f"Content: {result['chunk_text']}"
            for result in search_results[:5]
        ])
        
        # Contract analysis prompt
        contract_prompt = f"""
        As a legal expert specializing in contract analysis, please analyze the following contract provisions 
        in relation to this query: {request.query}
        
        Contract Context:
        {contract_context}
        
        Please provide:
        1. Contract clause analysis
        2. Risk assessment
        3. Compliance considerations
        4. Recommendations for negotiation or modification
        5. Standard vs. non-standard provisions identified
        
        Focus areas: {', '.join(request.focus_areas) if request.focus_areas else 'General analysis'}
        """
        
        # Use reasoning model for complex contract analysis
        response = together_client.chat.completions.create(
            model=os.getenv("REASONING_MODEL", "deepseek-ai/DeepSeek-R1"),
            messages=[
                {"role": "system", "content": "You are a senior contract attorney with expertise in commercial law."},
                {"role": "user", "content": contract_prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        return {
            "contract_analysis": response.choices[0].message.content,
            "sources_analyzed": len(search_results),
            "contract_type": request.contract_type,
            "focus_areas": request.focus_areas
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contract analysis failed: {str(e)}")

# Legal compliance check endpoint
@app.post("/compliance-check")
async def compliance_check(request: LegalAnalysisRequest):
    """Check legal compliance across documents"""
    
    try:
        # Search for relevant compliance documents
        search_results = legal_vector_store.search_legal_documents(
            query=f"compliance regulations {request.query}",
            top_k=8,
            document_type=request.document_types[0] if request.document_types else None
        )
        
        compliance_context = "\n\n".join([
            f"Document: {result['filename']}\n"
            f"Content: {result['chunk_text']}"
            for result in search_results[:5]
        ])
        
        compliance_prompt = f"""
        As a compliance expert, analyze the following documents for regulatory compliance:
        
        Query: {request.query}
        
        Document Context:
        {compliance_context}
        
        Please provide:
        1. Compliance status assessment
        2. Regulatory requirements identified
        3. Potential compliance gaps
        4. Recommended actions
        5. Risk level assessment (Low/Medium/High)
        """
        
        response = together_client.chat.completions.create(
            model=os.getenv("DEFAULT_LEGAL_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
            messages=[
                {"role": "system", "content": "You are a regulatory compliance expert."},
                {"role": "user", "content": compliance_prompt}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        return {
            "compliance_analysis": response.choices[0].message.content,
            "documents_reviewed": len(search_results),
            "analysis_type": request.analysis_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")

# Legal document summarization endpoint
@app.post("/document-summary")
async def document_summary(doc_id: int, summary_type: str = "comprehensive"):
    """Generate legal document summary"""
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get document content
        cur.execute("""
            SELECT filename, full_text, document_type, jurisdiction, practice_area, metadata
            FROM legal_documents 
            WHERE doc_id = %s
        """, (doc_id,))
        
        result = cur.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        
        filename, full_text, doc_type, jurisdiction, practice_area, metadata = result
        
        # Create summary prompt based on type
        if summary_type == "executive":
            prompt = f"""
            Provide an executive summary of this legal document:
            
            Document: {filename}
            Type: {doc_type}
            Jurisdiction: {jurisdiction}
            
            Content: {full_text[:3000]}...
            
            Include: Key points, main obligations, critical dates, and executive recommendations.
            """
        else:  # comprehensive
            prompt = f"""
            Provide a comprehensive legal analysis and summary of this document:
            
            Document: {filename}
            Type: {doc_type}
            
            Content: {full_text[:4000]}...
            
            Include: Detailed analysis, legal implications, key clauses, risks, and recommendations.
            """
        
        response = together_client.chat.completions.create(
            model=os.getenv("DEFAULT_LEGAL_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
            messages=[
                {"role": "system", "content": "You are an expert legal analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        cur.close()
        conn.close()
        
        return {
            "document_id": doc_id,
            "filename": filename,
            "document_type": doc_type,
            "summary_type": summary_type,
            "summary": response.choices[0].message.content,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document summary failed: {str(e)}")

# Legal search endpoint
@app.get("/search-documents")
async def search_legal_documents(
    query: str,
    document_type: Optional[str] = None,
    practice_area: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    limit: int = 10
):
    """Search legal documents with filters"""
    
    try:
        results = legal_vector_store.search_legal_documents(
            query=query,
            top_k=limit,
            document_type=document_type,
            practice_area=practice_area
        )
        
        return {
            "query": query,
            "total_results": len(results),
            "filters": {
                "document_type": document_type,
                "practice_area": practice_area,
                "jurisdiction": jurisdiction
            },
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Get document metadata endpoint
@app.get("/document/{doc_id}")
async def get_document_metadata(doc_id: int):
    """Get detailed document metadata"""
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT d.*, COUNT(c.chunk_id) as chunk_count
            FROM legal_documents d
            LEFT JOIN legal_chunks c ON d.doc_id = c.doc_id
            WHERE d.doc_id = %s
            GROUP BY d.doc_id
        """, (doc_id,))
        
        result = cur.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Convert result to dictionary
        columns = [desc[0] for desc in cur.description]
        doc_data = dict(zip(columns, result))
        
        cur.close()
        conn.close()
        
        return doc_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

# List all documents endpoint
@app.get("/documents")
async def list_documents(
    document_type: Optional[str] = None,
    practice_area: Optional[str] = None,
    limit: int = 50
):
    """List all legal documents with optional filters"""
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        query = """
            SELECT doc_id, filename, document_type, jurisdiction, practice_area, 
                   date_created, COUNT(c.chunk_id) as chunk_count
            FROM legal_documents d
            LEFT JOIN legal_chunks c ON d.doc_id = c.doc_id
        """
        
        conditions = []
        params = []
        
        if document_type:
            conditions.append("d.document_type = %s")
            params.append(document_type)
        
        if practice_area:
            conditions.append("d.practice_area = %s")
            params.append(practice_area)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " GROUP BY d.doc_id ORDER BY d.date_created DESC LIMIT %s"
        params.append(limit)
        
        cur.execute(query, params)
        results = cur.fetchall()
        
        columns = [desc[0] for desc in cur.description]
        documents = [dict(zip(columns, row)) for row in results]
        
        cur.close()
        conn.close()
        
        return {
            "total_documents": len(documents),
            "filters": {
                "document_type": document_type,
                "practice_area": practice_area
            },
            "documents": documents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)