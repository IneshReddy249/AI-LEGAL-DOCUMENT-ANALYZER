from io import BytesIO
import fitz  # PyMuPDF
import docx
import pdfplumber
from PIL import Image
import pytesseract
import re
from typing import Dict, List, Tuple

class LegalDocumentParser:
    def __init__(self):
        self.legal_patterns = {
            'contract_clauses': [
                r'WHEREAS\s+.*?;',
                r'NOW,?\s+THEREFORE\s+.*?;',
                r'IN\s+WITNESS\s+WHEREOF\s+.*?;'
            ],
            'legal_citations': [
                r'\d+\s+[A-Z][a-z]+\.?\s+\d+',  # Case citations
                r'\d+\s+U\.S\.C\.?\s+§?\s*\d+',  # USC citations
                r'\d+\s+C\.F\.R\.?\s+§?\s*\d+'   # CFR citations
            ],
            'dates': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ]
        }
    
    def extract_text_from_pdf(self, file_bytes: bytes) -> Dict:
        """Enhanced PDF extraction with legal document structure"""
        text = ""
        metadata = {
            'pages': 0,
            'has_images': False,
            'legal_elements': []
        }
        
        # Try pdfplumber first for better table extraction
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                metadata['pages'] = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}\n"
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            table_text = "\n".join(["\t".join(row) for row in table if row])
                            text += f"\n[TABLE]\n{table_text}\n[/TABLE]\n"
        except:
            # Fallback to PyMuPDF
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                metadata['pages'] = len(doc)
                for page_num, page in enumerate(doc, 1):
                    page_text = page.get_text().strip()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}\n"
        
        # Extract legal elements
        metadata['legal_elements'] = self._extract_legal_elements(text)
        
        return {
            'text': text.strip(),
            'metadata': metadata
        }
    
    def extract_text_from_docx(self, file_bytes: bytes) -> Dict:
        """Enhanced DOCX extraction with legal structure"""
        doc = docx.Document(BytesIO(file_bytes))
        text = ""
        metadata = {
            'paragraphs': 0,
            'tables': 0,
            'legal_elements': []
        }
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
                metadata['paragraphs'] += 1
        
        # Extract tables
        for table in doc.tables:
            metadata['tables'] += 1
            table_text = "\n[TABLE]\n"
            for row in table.rows:
                row_text = "\t".join([cell.text for cell in row.cells])
                table_text += row_text + "\n"
            table_text += "[/TABLE]\n"
            text += table_text
        
        metadata['legal_elements'] = self._extract_legal_elements(text)
        
        return {
            'text': text.strip(),
            'metadata': metadata
        }
    
    def _extract_legal_elements(self, text: str) -> List[Dict]:
        """Extract legal-specific elements from text"""
        elements = []
        
        for category, patterns in self.legal_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    elements.append({
                        'type': category,
                        'text': match.strip(),
                        'pattern': pattern
                    })
        
        return elements
    
    def extract_text(self, filename: str, file_bytes: bytes) -> Dict:
        """Main extraction method with enhanced legal processing"""
        if filename.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_bytes)
        elif filename.lower().endswith('.docx'):
            return self.extract_text_from_docx(file_bytes)
        elif filename.lower().endswith('.txt'):
            text = file_bytes.decode("utf-8", errors="ignore").strip()
            return {
                'text': text,
                'metadata': {
                    'legal_elements': self._extract_legal_elements(text)
                }
            }
        else:
            raise ValueError(f"Unsupported file type: {filename}")
