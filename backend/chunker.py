import re
from typing import List, Dict

class LegalChunker:
    def __init__(self):
        self.section_patterns = [
            r'^(?:ARTICLE|SECTION|CLAUSE)\s+[IVX\d]+',
            r'^\d+\.\s+[A-Z][^.]*\.',
            r'^[A-Z][A-Z\s]+:',  # All caps headers
        ]
    
    def chunk_legal_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Enhanced chunking that preserves legal document structure"""
        
        # First, try to split by legal sections
        sections = self._split_by_sections(text)
        
        if len(sections) > 1:
            return self._chunk_sections(sections, chunk_size, overlap)
        else:
            # Fallback to sentence-aware chunking
            return self._chunk_by_sentences(text, chunk_size, overlap)
    
    def _split_by_sections(self, text: str) -> List[Dict]:
        """Split text by legal sections/clauses"""
        sections = []
        current_section = ""
        current_title = "Introduction"
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            is_header = False
            for pattern in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_section.strip():
                        sections.append({
                            'title': current_title,
                            'content': current_section.strip(),
                            'type': 'section'
                        })
                    
                    # Start new section
                    current_title = line
                    current_section = ""
                    is_header = True
                    break
            
            if not is_header:
                current_section += line + "\n"
        
        # Add final section
        if current_section.strip():
            sections.append({
                'title': current_title,
                'content': current_section.strip(),
                'type': 'section'
            })
        
        return sections
    
    def _chunk_sections(self, sections: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
        """Chunk sections while preserving context"""
        chunks = []
        
        for section in sections:
            content = section['content']
            title = section['title']
            
            if len(content.split()) <= chunk_size:
                # Section fits in one chunk
                chunks.append({
                    'text': f"{title}\n\n{content}",
                    'type': 'legal_section',
                    'section_title': title,
                    'metadata': section
                })
            else:
                # Split large sections
                words = content.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    chunks.append({
                        'text': f"{title}\n\n{chunk_text}",
                        'type': 'legal_section_part',
                        'section_title': title,
                        'part_number': i // (chunk_size - overlap) + 1,
                        'metadata': section
                    })
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Sentence-aware chunking for better legal context"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_size + sentence_words > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'type': 'paragraph',
                    'word_count': current_size,
                    'metadata': {}
                })
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-1:]  # Keep last sentence
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_size = sentence_words
            else:
                current_chunk.append(sentence)
                current_size += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'type': 'paragraph',
                'word_count': current_size,
                'metadata': {}
            })
        
        return chunks