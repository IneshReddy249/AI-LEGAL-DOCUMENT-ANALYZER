from backend.db import get_db_connection
from backend.embeddings import LegalEmbeddings
from backend.chunker import LegalChunker
from backend.file_parser import LegalDocumentParser
from typing import List, Dict
import json
import os

class LegalVectorStore:
    def __init__(self):
        self.embeddings = LegalEmbeddings()
        self.chunker = LegalChunker()
        self.parser = LegalDocumentParser()
    
    def truncate_text(self, text: str, max_length: int) -> str:
        """Safely truncate text to fit database constraints"""
        if not text:
            return ""
        return text[:max_length] if len(text) > max_length else text
    
    def store_legal_document(self, filename: str, file_bytes: bytes, 
                           document_type: str = None, jurisdiction: str = None,
                           practice_area: str = None) -> Dict:
        """Store legal document with enhanced metadata"""
        
        # Parse document
        parsed_doc = self.parser.extract_text(filename, file_bytes)
        text = parsed_doc['text']
        metadata = parsed_doc['metadata']
        
        # Chunk the document
        chunks = self.chunker.chunk_legal_text(text)
        
        if not chunks:
            raise ValueError("No text chunks generated from document.")
        
        # Get embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.embeddings.get_embeddings_batch(chunk_texts)
        
        # Calculate document embedding (average of chunk embeddings)
        if chunk_embeddings:
            doc_embedding = [
                sum(chunk_embeddings[i][j] for i in range(len(chunk_embeddings))) / len(chunk_embeddings)
                for j in range(len(chunk_embeddings[0]))
            ]
        else:
            # Fallback: create embedding from first 1000 chars of text
            doc_embedding = self.embeddings.get_embedding(text[:1000])
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            # Truncate values to fit database constraints
            safe_filename = self.truncate_text(filename, 500)
            safe_doc_type = self.truncate_text(document_type, 200)
            safe_jurisdiction = self.truncate_text(jurisdiction, 200)
            safe_practice_area = self.truncate_text(practice_area, 200)
            
            # Insert document
            cur.execute("""
                INSERT INTO legal_documents 
                (filename, document_type, jurisdiction, practice_area, full_text, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING doc_id
            """, (
                safe_filename, safe_doc_type, safe_jurisdiction, safe_practice_area, 
                text, doc_embedding, json.dumps(metadata)
            ))
            
            doc_id = cur.fetchone()[0]
            
            # Insert chunks
            chunk_data = []
            for i, chunk in enumerate(chunks):
                if i < len(chunk_embeddings):  # Make sure we have embedding for this chunk
                    safe_chunk_type = self.truncate_text(chunk.get('type', 'paragraph'), 200)
                    safe_section_title = self.truncate_text(chunk.get('section_title', ''), 500)
                    
                    chunk_data.append((
                        doc_id,
                        chunk['text'],
                        safe_chunk_type,
                        safe_section_title,
                        chunk_embeddings[i],
                        chunk.get('part_number', 1)
                    ))
            
            if chunk_data:
                cur.executemany("""
                    INSERT INTO legal_chunks 
                    (doc_id, chunk_text, chunk_type, section_title, embedding, page_number)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, chunk_data)
            
            conn.commit()
            
            return {
                'doc_id': doc_id,
                'chunks_created': len(chunk_data),
                'document_type': safe_doc_type,
                'metadata': metadata
            }
            
        except Exception as e:
            conn.rollback()
            print(f"Database error: {e}")
            raise
        finally:
            cur.close()
            conn.close()
    
    def search_legal_documents(self, query: str, top_k: int = 5, 
                             document_type: str = None, 
                             practice_area: str = None) -> List[Dict]:
        """Enhanced search with legal filters"""
        
        query_embedding = self.embeddings.get_embedding(query)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Build query with optional filters
        base_query = """
            SELECT 
                c.chunk_id, c.doc_id, c.chunk_text, c.chunk_type, c.section_title,
                d.filename, d.document_type, d.jurisdiction, d.practice_area,
                (c.embedding <#> %s::vector) AS similarity_score
            FROM legal_chunks c
            JOIN legal_documents d ON c.doc_id = d.doc_id
        """
        
        conditions = []
        params = [query_embedding]
        
        if document_type:
            conditions.append("d.document_type = %s")
            params.append(document_type)
        
        if practice_area:
            conditions.append("d.practice_area = %s")
            params.append(practice_area)
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        base_query += " ORDER BY similarity_score ASC LIMIT %s"
        params.append(top_k)
        
        cur.execute(base_query, params)
        rows = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return [
            {
                'chunk_id': row[0],
                'doc_id': row[1],
                'chunk_text': row[2],
                'chunk_type': row[3],
                'section_title': row[4],
                'filename': row[5],
                'document_type': row[6],
                'jurisdiction': row[7],
                'practice_area': row[8],
                'similarity_score': float(row[9])
            }
            for row in rows
        ]
