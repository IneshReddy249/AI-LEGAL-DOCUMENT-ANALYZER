import os
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector

load_dotenv()

def get_db_connection():
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
    """Create enhanced tables for legal document analysis"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Enhanced documents table with legal metadata
    cur.execute("""
        CREATE TABLE IF NOT EXISTS legal_documents (
            doc_id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            document_type VARCHAR(100), -- contract, legal_brief, statute, etc.
            jurisdiction VARCHAR(100),
            practice_area VARCHAR(100), -- corporate, litigation, etc.
            date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            full_text TEXT,
            embedding VECTOR(1536),
            metadata JSONB -- store additional legal metadata
        );
    """)
    
    # Enhanced chunks table with legal context
    cur.execute("""
        CREATE TABLE IF NOT EXISTS legal_chunks (
            chunk_id SERIAL PRIMARY KEY,
            doc_id INTEGER REFERENCES legal_documents(doc_id),
            chunk_text TEXT NOT NULL,
            chunk_type VARCHAR(50), -- clause, paragraph, section, etc.
            legal_concepts TEXT[], -- array of identified legal concepts
            embedding VECTOR(1536),
            page_number INTEGER,
            section_title VARCHAR(255)
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    try:
        create_legal_tables()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print(f"✅ PostgreSQL connected successfully! Version: {db_version[0]}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Failed to connect. Error: {e}")