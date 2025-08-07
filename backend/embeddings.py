import os
from together import Together
from typing import List
import re

class LegalEmbeddings:
    def __init__(self):
        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        print(f"Initialized embeddings with model: {self.embedding_model}")
    
    def clean_text(self, text: str) -> str:
        """Aggressively clean text for embedding API"""
        if not text or not text.strip():
            return "Empty text placeholder"
        
        # Remove non-printable characters except common whitespace
        clean_text = ''.join(char for char in text if char.isprintable() or char in [' ', '\n', '\t'])
        
        # Remove excessive whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Remove special characters that might cause issues
        clean_text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]{}"\']', ' ', clean_text)
        
        # Trim and ensure not empty
        clean_text = clean_text.strip()
        if not clean_text:
            return "Empty text placeholder"
        
        # Limit length (conservative limit)
        if len(clean_text) > 3000:
            clean_text = clean_text[:3000]
        
        return clean_text
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text with aggressive cleaning"""
        try:
            clean_text = self.clean_text(text)
            
            response = self.client.embeddings.create(
                input=[clean_text],
                model=self.embedding_model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"❌ Single embedding failed for text: {repr(text[:100])}")
            print(f"❌ Cleaned text: {repr(self.clean_text(text)[:100])}")
            print(f"❌ Error: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with individual calls and better error handling"""
        try:
            print(f"=== BATCH EMBEDDING (Individual Calls with Cleaning) ===")
            print(f"Number of texts: {len(texts)}")
            
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    print(f"Processing text {i+1}/{len(texts)}")
                    
                    # Clean the text aggressively
                    clean_text = self.clean_text(text)
                    
                    # Skip if text becomes too short after cleaning
                    if len(clean_text) < 10:
                        print(f"Warning: Text {i+1} too short after cleaning, using placeholder")
                        clean_text = f"Document section {i+1} placeholder text"
                    
                    # Get individual embedding
                    embedding = self.get_embedding(clean_text)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    print(f"❌ Failed on text {i+1}: {e}")
                    print(f"❌ Problematic text: {repr(text[:200])}")
                    
                    # Use a safe placeholder embedding
                    try:
                        placeholder_embedding = self.get_embedding(f"Placeholder text for chunk {i+1}")
                        embeddings.append(placeholder_embedding)
                        print(f"✅ Used placeholder for text {i+1}")
                    except:
                        # If even placeholder fails, skip this chunk
                        print(f"❌ Skipping text {i+1} entirely")
                        continue
            
            print(f"✅ Batch embedding completed: {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            print(f"❌ Batch embedding failed: {e}")
            raise
