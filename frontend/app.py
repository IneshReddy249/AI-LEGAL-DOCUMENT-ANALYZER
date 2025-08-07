import gradio as gr
import requests
import os
import json

API_URL = "http://localhost:8000"

def upload_legal_doc(file, doc_type, jurisdiction, practice_area):
    """Upload legal document with metadata"""
    if file is None:
        return "❌ Please select a file."
    
    try:
        # Handle different Gradio versions - FIXED VERSION
        if hasattr(file, 'name'):
            # Newer Gradio versions - file object with .name attribute
            file_path = file.name
            filename = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                file_content = f.read()
        elif isinstance(file, str):
            # Older Gradio versions - file path as string
            file_path = file
            filename = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                file_content = f.read()
        else:
            # Fallback - try to read directly
            filename = getattr(file, 'orig_name', 'unknown_file')
            file_content = file.read() if hasattr(file, 'read') else file
        
        # Prepare request exactly like working curl command
        files = {'file': (filename, file_content)}
        data = {
            'document_type': doc_type,
            'jurisdiction': jurisdiction,
            'practice_area': practice_area
        }
        
        print(f"Uploading: {filename}, size: {len(file_content)} bytes")
        
        r = requests.post(f"{API_URL}/upload-and-store", files=files, data=data)
        
        print(f"Response status: {r.status_code}")
        print(f"Response: {r.text}")
        
        if r.status_code == 200:
            result = r.json()
            return f"✅ Document uploaded successfully!\n\n" \
                   f"Document ID: {result['doc_id']}\n" \
                   f"Chunks created: {result['chunks_created']}\n" \
                   f"Document type: {result['document_type']}\n" \
                   f"Filename: {result['filename']}\n" \
                   f"Message: {result['message']}\n\n" \
                   f"Metadata: {json.dumps(result['metadata'], indent=2)}"
        else:
            return f"❌ Upload error: {r.text}"
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Upload failed: {e}\n\nDetails:\n{error_details}"

def legal_rag_query(question, top_k, doc_type, practice_area):
    """Enhanced legal RAG query"""
    if not question.strip():
        return "❌ Please enter a question."
    
    payload = {
        "query": question,
        "top_k": int(top_k),
        "document_type": doc_type if doc_type != "Any" else None,
        "practice_area": practice_area if practice_area != "Any" else None
    }
    
    try:
        r = requests.post(f"{API_URL}/rag-answer", json=payload)
        
        if r.status_code == 200:
            result = r.json()
            
            # Format response
            response = f"**Question:** {question}\n\n"
            response += f"**Legal Analysis:**\n{result['answer']}\n\n"
            response += f"**Confidence Score:** {result['confidence_score']:.2f}\n\n"
            
            # Legal analysis summary
            analysis = result['legal_analysis']
            response += f"**Analysis Summary:**\n"
            response += f"- Documents analyzed: {analysis['total_sources']}\n"
            response += f"- Document types: {', '.join(analysis['document_types_analyzed'])}\n"
            response += f"- Practice areas: {', '.join(analysis['practice_areas'])}\n"
            response += f"- Average relevance: {analysis['avg_relevance']:.3f}\n\n"
            
            # Sources
            response += f"**Sources:**\n"
            for idx, source in enumerate(result['sources'][:3], 1):
                response += f"\n--- Source {idx} ---\n"
                response += f"Document: {source['source']}\n"
                response += f"Type: {source['type']}\n"
                response += f"Section: {source['section']}\n"
                response += f"Relevance: {source['score']:.3f}\n"
                response += f"Content: {source['text'][:300]}...\n"
            
            return response
        else:
            return f"❌ Error: {r.text}"
    except Exception as e:
        return f"❌ Failed to contact backend: {e}"

def contract_analysis(question, contract_type, focus_areas):
    """Contract-specific analysis"""
    if not question.strip():
        return "❌ Please enter a question."
    
    payload = {
        "query": question,
        "contract_type": contract_type if contract_type != "Any" else None,
        "focus_areas": [area.strip() for area in focus_areas.split(",")] if focus_areas else None
    }
    
    try:
        r = requests.post(f"{API_URL}/contract-analysis", json=payload)
        
        if r.status_code == 200:
            result = r.json()
            return f"**Contract Analysis:**\n\n{result['contract_analysis']}\n\n" \
                   f"**Sources Analyzed:** {result['sources_analyzed']}"
        else:
            return f"❌ Error: {r.text}"
    except Exception as e:
        return f"❌ Failed to contact backend: {e}"

def compliance_check(question, doc_types):
    """Legal compliance check"""
    if not question.strip():
        return "❌ Please enter a compliance question."
    
    payload = {
        "query": question,
        "document_types": [doc_types] if doc_types != "Any" else None,
        "analysis_type": "compliance"
    }
    
    try:
        r = requests.post(f"{API_URL}/compliance-check", json=payload)
        
        if r.status_code == 200:
            result = r.json()
            return f"**Compliance Analysis:**\n\n{result['compliance_analysis']}\n\n" \
                   f"**Documents Reviewed:** {result['documents_reviewed']}"
        else:
            return f"❌ Error: {r.text}"
    except Exception as e:
        return f"❌ Failed to contact backend: {e}"

# Gradio Interface
with gr.Blocks(title="🧑‍⚖️ AI Legal Document Analyzer") as demo:
    gr.Markdown("# 🧑‍⚖️ AI Legal Document Analyzer")
    gr.Markdown("Advanced legal document analysis powered by Together AI")
    gr.Markdown("✅ **Status: File handling fixed and API verified working!**")
    
    with gr.Tab("📄 Upload Legal Document"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload Legal Document")
                doc_type = gr.Dropdown(
                    choices=["contract", "legal_brief", "statute", "regulation", "case_law", "other"],
                    label="Document Type",
                    value="contract"
                )
                jurisdiction = gr.Textbox(label="Jurisdiction", placeholder="e.g., New York, Federal, EU", value="US")
                practice_area = gr.Dropdown(
                    choices=["corporate", "litigation", "employment", "real_estate", "intellectual_property", "other"],
                    label="Practice Area",
                    value="corporate"
                )
                upload_btn = gr.Button("🚀 Upload and Process", variant="primary")
            
            with gr.Column():
                upload_output = gr.Textbox(label="Upload Status", interactive=False, lines=12)
        
        upload_btn.click(
            upload_legal_doc,
            inputs=[file_input, doc_type, jurisdiction, practice_area],
            outputs=[upload_output]
        )
    
    with gr.Tab("🔍 Legal Q&A"):
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(
                    label="Legal Question",
                    placeholder="What are the liability provisions in the contract?",
                    lines=3
                )
                top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Sources")
                doc_type_filter = gr.Dropdown(
                    choices=["Any", "contract", "legal_brief", "statute", "regulation", "case_law"],
                    label="Filter by Document Type",
                    value="Any"
                )
                practice_area_filter = gr.Dropdown(
                    choices=["Any", "corporate", "litigation", "employment", "real_estate", "intellectual_property"],
                    label="Filter by Practice Area",
                    value="Any"
                )
                ask_btn = gr.Button("🔍 Analyze", variant="primary")
            
            with gr.Column():
                answer_output = gr.Markdown()
        
        ask_btn.click(
            legal_rag_query,
            inputs=[question, top_k, doc_type_filter, practice_area_filter],
            outputs=[answer_output]
        )
    
    with gr.Tab("📋 Contract Analysis"):
        with gr.Row():
            with gr.Column():
                contract_question = gr.Textbox(
                    label="Contract Analysis Question",
                    placeholder="Analyze the termination clauses and associated risks",
                    lines=3
                )
                contract_type_input = gr.Dropdown(
                    choices=["Any", "employment", "service", "purchase", "lease", "nda", "partnership"],
                    label="Contract Type",
                    value="Any"
                )
                focus_areas_input = gr.Textbox(
                    label="Focus Areas (comma-separated)",
                    placeholder="liability, termination, payment, intellectual_property"
                )
                contract_btn = gr.Button("📋 Analyze Contract", variant="primary")
            
            with gr.Column():
                contract_output = gr.Markdown()
        
        contract_btn.click(
            contract_analysis,
            inputs=[contract_question, contract_type_input, focus_areas_input],
            outputs=[contract_output]
        )
    
    with gr.Tab("✅ Compliance Check"):
        with gr.Row():
            with gr.Column():
                compliance_question = gr.Textbox(
                    label="Compliance Question",
                    placeholder="Check GDPR compliance in our data processing agreements",
                    lines=3
                )
                compliance_doc_type = gr.Dropdown(
                    choices=["Any", "contract", "policy", "regulation", "procedure"],
                    label="Document Type",
                    value="Any"
                )
                compliance_btn = gr.Button("✅ Check Compliance", variant="primary")
            
            with gr.Column():
                compliance_output = gr.Markdown()
        
        compliance_btn.click(
            compliance_check,
            inputs=[compliance_question, compliance_doc_type],
            outputs=[compliance_output]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)