import re
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from backend.embeddings import LegalEmbeddings

class SemanticChunker:
    def __init__(self, similarity_threshold: float = 0.7, max_chunk_size: int = 800):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.embeddings = LegalEmbeddings()
        
    def semantic_chunk_legal_text(self, text: str) -> List[Dict]:
        """Enhanced semantic chunking for legal documents"""
        
        # First split by legal sections if possible
        sections = self._split_by_legal_sections(text)
        
        all_chunks = []
        for section in sections:
            # Apply semantic chunking within each section
            semantic_chunks = self._semantic_chunk_section(section)
            all_chunks.extend(semantic_chunks)
        
        return all_chunks
    
    def _split_by_legal_sections(self, text: str) -> List[Dict]:
        """Split text by legal sections first"""
        section_patterns = [
            r'^(?:ARTICLE|SECTION|CLAUSE)\s+[IVX\d]+',
            r'^\d+\.\s+[A-Z][^.]*\.',
            r'^[A-Z][A-Z\s]+:',
            r'^WHEREAS\s+.*?;',
            r'^NOW,?\s+THEREFORE\s+.*?;'
        ]
        
        sections = []
        current_section = {"title": "Introduction", "content": "", "start_pos": 0}
        
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                current_section["content"] += "\n"
                continue
                
            is_section_header = False
            for pattern in section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    if current_section["content"].strip():
                        current_section["end_pos"] = current_pos
                        sections.append(current_section)
                    
                    current_section = {
                        "title": line,
                        "content": "",
                        "start_pos": current_pos,
                        "type": "legal_section"
                    }
                    is_section_header = True
                    break
            
            if not is_section_header:
                current_section["content"] += line + "\n"
            
            current_pos += len(line) + 1
        
        if current_section["content"].strip():
            current_section["end_pos"] = current_pos
            sections.append(current_section)
        
        return sections if len(sections) > 1 else [{"title": "Document", "content": text, "type": "full_document"}]
    
    def _semantic_chunk_section(self, section: Dict) -> List[Dict]:
        """Apply semantic chunking within a section"""
        content = section["content"]
        
        sentences = self._split_into_sentences(content)
        if len(sentences) <= 1:
            return [{
                "text": content,
                "type": "semantic_chunk",
                "section_title": section["title"],
                "sentence_count": len(sentences)
            }]
        
        sentence_embeddings = self.embeddings.get_embeddings_batch(sentences)
        
        chunks = self._group_sentences_by_similarity(sentences, sentence_embeddings, section["title"])
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with legal document awareness"""
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_endings, text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _group_sentences_by_similarity(self, sentences: List[str], embeddings: List[List[float]], section_title: str) -> List[Dict]:
        """Group sentences into chunks based on semantic similarity"""
        if not sentences or not embeddings:
            return []
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embeddings = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            chunk_avg_embedding = np.mean(current_chunk_embeddings, axis=0)
            similarity = cosine_similarity([embeddings[i]], [chunk_avg_embedding])[0][0]
            
            current_chunk_text = " ".join(current_chunk_sentences + [sentences[i]])
            
            if (similarity >= self.similarity_threshold and 
                len(current_chunk_text) <= self.max_chunk_size):
                current_chunk_sentences.append(sentences[i])
                current_chunk_embeddings.append(embeddings[i])
            else:
                chunks.append({
                    "text": " ".join(current_chunk_sentences),
                    "type": "semantic_chunk",
                    "section_title": section_title,
                    "sentence_count": len(current_chunk_sentences),
                    "avg_similarity": np.mean([cosine_similarity([current_chunk_embeddings[j]], [current_chunk_embeddings[0]])[0][0] for j in range(1, len(current_chunk_embeddings))]) if len(current_chunk_embeddings) > 1 else 1.0
                })
                
                current_chunk_sentences = [sentences[i]]
                current_chunk_embeddings = [embeddings[i]]
        
        if current_chunk_sentences:
            chunks.append({
                "text": " ".join(current_chunk_sentences),
                "type": "semantic_chunk",
                "section_title": section_title,
                "sentence_count": len(current_chunk_sentences),
                "avg_similarity": np.mean([cosine_similarity([current_chunk_embeddings[j]], [current_chunk_embeddings[0]])[0][0] for j in range(1, len(current_chunk_embeddings))]) if len(current_chunk_embeddings) > 1 else 1.0
            })
        
        return chunks