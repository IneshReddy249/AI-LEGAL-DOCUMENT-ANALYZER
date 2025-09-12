import os
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector

load_dotenv()

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        database=os.getenv("DB_NAME", "legal_db"),
        user=os.getenv("DB_USER", "legal_user"),
        password=os.getenv("DB_PASSWORD", "")
    )
    register_vector(conn)
    return conn

def create_legal_tables():
    """
    Create or update tables for the legal document analysis system,
    optimized for semantic chunking and hybrid search.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get embedding dimension from env var, defaulting to 768 for BAAI/bge-base-en-v1.5
    embedding_dim = os.getenv("EMBEDDING_DIM", 768)
    
    # Enable the vector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Enhanced documents table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS legal_documents (
            doc_id SERIAL PRIMARY KEY,
            filename VARCHAR(500) NOT NULL,
            document_type VARCHAR(200),
            jurisdiction VARCHAR(200),
            practice_area VARCHAR(200),
            date_created TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            full_text TEXT,
            embedding VECTOR(%(embedding_dim)s),
            metadata JSONB
        );
    """, {'embedding_dim': embedding_dim})
    
    # Enhanced chunks table, optimized for hybrid search
    cur.execute("""
        CREATE TABLE IF NOT EXISTS legal_chunks (
            chunk_id SERIAL PRIMARY KEY,
            doc_id INTEGER REFERENCES legal_documents(doc_id) ON DELETE CASCADE,
            chunk_text TEXT NOT NULL,
            chunk_type VARCHAR(200),
            section_title VARCHAR(500),
            embedding VECTOR(%(embedding_dim)s),
            sentence_count INTEGER, -- Renamed from page_number for clarity
            chunk_metadata JSONB, -- For extra info like avg_similarity
            tsv TSVECTOR -- For full-text search (BM25 approximation)
        );
    """, {'embedding_dim': embedding_dim})
    
    # Trigger to automatically update the tsv column for full-text search
    cur.execute("""
        CREATE OR REPLACE FUNCTION update_tsv_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.tsv := to_tsvector('english', NEW.chunk_text);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    cur.execute("""
        DROP TRIGGER IF EXISTS tsvector_update ON legal_chunks;
        CREATE TRIGGER tsvector_update
        BEFORE INSERT OR UPDATE ON legal_chunks
        FOR EACH ROW EXECUTE FUNCTION update_tsv_column();
    """)
    
    # --- INDEXING FOR PERFORMANCE ---
    # HNSW index for fast vector similarity search
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_legal_chunks_embedding
        ON legal_chunks
        USING hnsw (embedding vector_l2_ops);
    """)
    
    # GIN index for fast full-text search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_legal_chunks_tsv
        ON legal_chunks
        USING GIN (tsv);
    """)
    
    # Index on foreign key for faster joins
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_legal_chunks_doc_id
        ON legal_chunks (doc_id);
    """)
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    try:
        print("Initializing database schema...")
        create_legal_tables()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print(f"✅ PostgreSQL connected successfully! Version: {db_version[0]}")
        print("✅ Database schema initialized/updated successfully.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Database initialization failed. Error: {e}")