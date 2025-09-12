# frontend/app.py
import os
import datetime as dt
from typing import Any, List
import requests
import gradio as gr

DEFAULT_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# ---------- Base structural CSS (uses variables that themes override) ----------
BASE_CSS = """
:root{
  --bg:#f2f4f8; --panel:#ffffff; --line:#e5e7eb;
  --ink:#0f172a; --muted:#475569;
  --brand:#6366f1; --brand-2:#4f46e5; --accent:#22c55e; --warn:#ef4444; --chip:#eef2ff;
}
html, body, .gradio-container{
  background:var(--bg)!important; color:var(--ink)!important;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
}
.gradio-container{
  max-width:1180px!important; margin:0 auto!important;
  font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,'Helvetica Neue',Arial,sans-serif!important;
  font-size:15px;
}
.gradio-container, .gradio-container * { color: var(--ink); }
.label, .wrap .label, .contain .label { color:var(--muted)!important; font-weight:800; letter-spacing:.02em; }

/* Headline */
.hero{
  background:var(--panel); border:1px solid var(--line);
  border-radius:16px; padding:22px 24px; margin:20px 0;
}
.hero h1{ margin:0 0 6px 0; font-size:28px; letter-spacing:-.01em; font-weight:900; }
.hero p{ margin:0; color:var(--muted)!important; }

/* KPI cards */
.kpi-grid{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
.kpi-card{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }
.kpi-title{ font-size:12.5px; color:var(--muted); font-weight:900; letter-spacing:.03em; text-transform:uppercase; }
.kpi-value{ margin-top:6px; font-size:22px; font-weight:900; color:var(--ink); }

/* Inputs / Buttons */
.input-field .wrap, .input-field textarea, .input-field input, .input-field select{
  background:var(--panel)!important; color:var(--ink)!important;
  border:1px solid var(--line)!important; border-radius:10px!important;
}
.primary-button button{
  background:var(--brand)!important; border:none!important; color:#fff!important;
  font-weight:900!important; border-radius:10px!important; letter-spacing:.02em;
}
.primary-button button:hover{ background:var(--brand-2)!important; }

/* Panels & Results */
.panel{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }
.results{
  background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--brand);
  border-radius:12px; padding:18px;
}

/* Dataframe */
.dataframe-wrap{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:6px; }
.dataframe thead th { background: var(--chip)!important; color: var(--ink)!important; font-weight:900!important; }
.dataframe td, .dataframe th { border-color: var(--line)!important; font-size:14px; }

/* Tabs underline */
.tabs > .tab-nav button[aria-selected="true"]{
  border-bottom:2px solid var(--brand)!important; color:var(--brand)!important; font-weight:900;
}

/* Footer */
.footer{ text-align:center; padding:22px 0; color:var(--muted); }
.footer a{ color:var(--brand)!important; font-weight:700; }
"""

# ---------- Theme variable sets ----------
# 1) Light (High Contrast)
THEME_LIGHT_VARS = """
<style id="theme-vars">
:root{
  --bg:#f3f4f6; --panel:#ffffff; --line:#e5e7eb;
  --ink:#0b1220; --muted:#425466;
  --brand:#5865f2; --brand-2:#4854e6; --accent:#22c55e; --warn:#ef4444; --chip:#eef2ff;
}
</style>
"""

# 2) Dark (High Contrast)
THEME_DARK_VARS = """
<style id="theme-vars">
:root{
  --bg:#0b1220; --panel:#121a2a; --line:#2a3345;
  --ink:#f5f7fa; --muted:#cbd5e1;
  --brand:#7c83ff; --brand-2:#5b61ff; --accent:#34d399; --warn:#f87171; --chip:#1b2232;
}
</style>
"""

# 3) Dim / Neutral (NOT light, NOT dark) – balanced contrast
THEME_DIM_VARS = """
<style id="theme-vars">
:root{
  /* Neutral slate with comfortable contrast */
  --bg:#1b2332;        /* deep slate (not black) */
  --panel:#242f41;     /* slightly lighter panels */
  --line:#3a4961;      /* soft borders */
  --ink:#eaf0f7;       /* high-legibility text */
  --muted:#c9d3df;     /* secondary text */
  --brand:#8aa4ff;     /* calm indigo */
  --brand-2:#6f8cff;
  --accent:#4ad4a0;
  --warn:#ff7a7a;
  --chip:#2b384f;      /* table header bg */
}
</style>
"""

# ---------- Helpers ----------
def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def kpi_card(title: str, value: str | int) -> str:
    return f'<div class="kpi-card"><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div></div>'

def safe_filename(path: str) -> str:
    return os.path.basename(path or "document")

def format_row(doc_id: int, filename: str, doc_type: str, practice: str, chunks: int):
    return [doc_id, filename, doc_type or "—", practice or "—", chunks, now_str()]

# ---------- Backend wrappers ----------
def upload_legal_doc(file, doc_type: str, practice_area: str,
                     api_url: str, table: List[List[Any]], docs_count: int, chunks_total: int, last_uploaded: str):
    api_url = (api_url or DEFAULT_API_URL).strip() or DEFAULT_API_URL
    if file is None:
        return ("❌ Please select a file to upload.", table, docs_count, chunks_total, last_uploaded,
                kpi_card("Documents this session", docs_count or 0),
                kpi_card("Total chunks indexed", chunks_total or 0),
                kpi_card("Last uploaded", last_uploaded or "—"))
    try:
        filename = safe_filename(file.name)
        with open(file.name, "rb") as f:
            file_bytes = f.read()
        files = {"file": (filename, file_bytes)}
        data = {"document_type": doc_type, "practice_area": practice_area}
        r = requests.post(f"{api_url}/upload-and-store", files=files, data=data, timeout=180)
        if r.status_code != 200:
            return (f"❌ **Upload Error:** {r.status_code} - {r.text}", table, docs_count, chunks_total, last_uploaded,
                    kpi_card("Documents this session", docs_count or 0),
                    kpi_card("Total chunks indexed", chunks_total or 0),
                    kpi_card("Last uploaded", last_uploaded or "—"))
        res = r.json()
        doc_id = res.get("doc_id")
        chunks = int(res.get("chunks_created", 0))
        shown_filename = res.get("filename", filename)
        method = res.get("chunking_method", "semantic")
        new_table = (table or []) + [format_row(doc_id, shown_filename, doc_type, practice_area, chunks)]
        new_docs = (docs_count or 0) + 1
        new_chunks = (chunks_total or 0) + chunks
        new_last = now_str()
        md = f"""
## ✅ Document processed

- **Document ID:** `{doc_id}`
- **Filename:** `{shown_filename}`
- **Type / Practice:** `{doc_type}` / `{practice_area}`
- **Chunks created:** `{chunks}`
- **Chunking:** `{method}`

Use **Legal Q&A** to ask questions, or **Summaries** to generate a brief.
"""
        return (md, new_table, new_docs, new_chunks, new_last,
                kpi_card("Documents this session", new_docs),
                kpi_card("Total chunks indexed", new_chunks),
                kpi_card("Last uploaded", new_last))
    except Exception as e:
        return (f"❌ **Error:** {str(e)}", table, docs_count, chunks_total, last_uploaded,
                kpi_card("Documents this session", docs_count or 0),
                kpi_card("Total chunks indexed", chunks_total or 0),
                kpi_card("Last uploaded", last_uploaded or "—"))

def agentic_rag_query(question: str, top_k: int, doc_type: str, practice_area: str, api_url: str):
    api_url = (api_url or DEFAULT_API_URL).strip() or DEFAULT_API_URL
    if not question or not question.strip():
        return "❌ Please enter a question."
    payload = {
        "query": question.strip(),
        "top_k": int(top_k),
        "document_type": None if doc_type == "Any" else doc_type,
        "practice_area": None if practice_area == "Any" else practice_area,
    }
    try:
        r = requests.post(f"{api_url}/agentic-rag-answer", json=payload, timeout=120)
        if r.status_code != 200:
            return f"❌ **Error:** {r.status_code} - {r.text}"
        result = r.json()
        conf = result.get("confidence_score", 0.0)
        answer = result.get("answer", "—")
        uniq, seen = [], set()
        for s in result.get("sources", []):
            txt = s.get("text", "")
            if txt in seen: continue
            seen.add(txt); uniq.append(s)
        md = [f"## 🤖 AI Legal Analysis\n\n**Question:** {question}\n\n---\n\n### 📋 Answer\n\n{answer}\n\n---"]
        md.append(f"### 📊 Analysis\n- **Confidence:** `{conf:.1%}`\n- **Search:** `Hybrid + Rerank`\n- **Sources used:** `{len(uniq)}`\n")
        if uniq:
            md.append("### 📚 Top sources\n")
            for i, s in enumerate(uniq[:3], 1):
                src = s.get("source", "N/A"); rel = s.get("rerank_score", 0.0)
                prev = (s.get("text","") or "").strip().replace("\n"," ")
                if len(prev) > 220: prev = prev[:220] + "..."
                md.append(f"**Source {i}:** `{src}`  \n**Relevance:** `{rel:.1%}`  \n**Preview:** {prev}\n")
        return "\n".join(md)
    except Exception as e:
        return f"❌ **Connection Error:** {str(e)}"

def generate_summary(doc_id: Any, summary_type: str, api_url: str):
    api_url = (api_url or DEFAULT_API_URL).strip() or DEFAULT_API_URL
    if doc_id in (None, "", " "):
        return "❌ Please enter a Document ID."
    try:
        params = {"doc_id": int(doc_id), "summary_type": summary_type}
        r = requests.post(f"{api_url}/document-summary", params=params, timeout=120)
        if r.status_code != 200:
            return f"❌ **Error:** {r.status_code} - {r.text}"
        res = r.json()
        stamp = (res.get("generated_at") or "").replace("T", " ")[:19]
        return f"""## Summary — {summary_type.title()}

**Document ID:** `{res.get('document_id')}`  
**Filename:** `{res.get('filename','—')}`  
**Generated:** `{stamp or now_str()}`

---

{res.get('summary','—')}

---
*AI-generated from the full document content.*"""
    except Exception as e:
        return f"❌ **Error:** {str(e)}"

def set_api_url(new_url: str):
    new_url = (new_url or "").strip()
    if not new_url:
        return DEFAULT_API_URL, f"Using default API URL: `{DEFAULT_API_URL}`"
    return new_url, f"API URL set to: `{new_url}`"

def apply_theme(choice: str):
    if choice == "Dark (High Contrast)":
        return THEME_DARK_VARS, f"Theme applied: **{choice}**"
    if choice == "Light (High Contrast)":
        return THEME_LIGHT_VARS, f"Theme applied: **{choice}**"
    # Default to the requested neutral option
    return THEME_DIM_VARS, f"Theme applied: **{choice}**"

# ---------- UI ----------
with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="indigo",
        neutral_hue="gray",
        font=gr.themes.GoogleFont("Inter"),
        radius_size=gr.themes.sizes.radius_sm,
    ),
    css=BASE_CSS,
    title="AI Legal Document Analyzer — Demo",
) as demo:

    # States
    st_api_url = gr.State(DEFAULT_API_URL)
    st_table = gr.State([])     # rows: [doc_id, filename, type, practice, chunks, added]
    st_docs = gr.State(0)
    st_chunks = gr.State(0)
    st_last = gr.State("—")

    # Theme slot (default to Dim / Neutral)
    theme_slot = gr.HTML(THEME_DIM_VARS)

    # Header
    gr.HTML('<div class="hero"><h1>AI Legal Document Analyzer</h1><p>Upload → Ask → Summarize. Clear, source-grounded answers from your legal documents.</p></div>')

    # KPIs
    with gr.Row(elem_classes="kpi-grid"):
        kpi_docs = gr.HTML(kpi_card("Documents this session", 0))
        kpi_chunks = gr.HTML(kpi_card("Total chunks indexed", 0))
        kpi_last = gr.HTML(kpi_card("Last uploaded", "—"))

    with gr.Tabs():

        # Overview
        with gr.Tab("Overview"):
            gr.Markdown("Use the tabs above. Start with **Upload**, then go to **Legal Q&A** or **Summaries**.", elem_classes="panel")
            overview_table = gr.Dataframe(
                headers=["Document ID", "Filename", "Type", "Practice area", "Chunks", "Added"],
                value=[], interactive=False, wrap=True, elem_classes="dataframe-wrap"
            )

        # Upload
        with gr.Tab("Upload"):
            gr.Markdown("### 📄 Upload legal document", elem_classes="panel")
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Select document", file_types=[".pdf", ".docx", ".txt"], elem_classes="input-field")
                    doc_type = gr.Dropdown(
                        choices=["contract", "legal_brief", "statute", "regulation", "case_law", "other"],
                        value="contract", label="Document type", elem_classes="input-field")
                    practice_area = gr.Dropdown(
                        choices=["corporate", "litigation", "employment", "real_estate", "intellectual_property", "other"],
                        value="corporate", label="Practice area", elem_classes="input-field")
                    upload_btn = gr.Button("Process document", elem_classes="primary-button")
                with gr.Column(scale=2):
                    upload_out = gr.Markdown(elem_classes="results")

            upload_btn.click(
                upload_legal_doc,
                inputs=[file_input, doc_type, practice_area, st_api_url, st_table, st_docs, st_chunks, st_last],
                outputs=[upload_out, st_table, st_docs, st_chunks, st_last, kpi_docs, kpi_chunks, kpi_last],
            ).then(lambda tbl: tbl, inputs=[st_table], outputs=[overview_table])

        # Q&A
        with gr.Tab("Legal Q&A"):
            gr.Markdown("### 💬 Ask questions about your uploaded documents", elem_classes="panel")
            with gr.Row():
                with gr.Column(scale=1):
                    q_box = gr.Textbox(
                        label="Your legal question",
                        placeholder="e.g., What are the termination clauses in this contract?",
                        lines=4, elem_classes="input-field")
                    top_k = gr.Slider(1, 10, value=5, step=1, label="Number of sources", elem_classes="input-field")
                    with gr.Row():
                        dt_filter = gr.Dropdown(
                            choices=["Any", "contract", "legal_brief", "statute", "regulation", "case_law"],
                            value="Any", label="Filter by document type", elem_classes="input-field")
                        pa_filter = gr.Dropdown(
                            choices=["Any", "corporate", "litigation", "employment", "real_estate", "intellectual_property"],
                            value="Any", label="Filter by practice area", elem_classes="input-field")
                    ask_btn = gr.Button("Get AI analysis", elem_classes="primary-button")
                with gr.Column(scale=2):
                    q_out = gr.Markdown(elem_classes="results")
            ask_btn.click(agentic_rag_query, inputs=[q_box, top_k, dt_filter, pa_filter, st_api_url], outputs=[q_out])

        # Summaries
        with gr.Tab("Summaries"):
            gr.Markdown("### 📋 Generate document summary", elem_classes="panel")
            with gr.Row():
                with gr.Column(scale=1):
                    sum_doc_id = gr.Number(label="Document ID", elem_classes="input-field")
                    sum_type = gr.Dropdown(choices=["comprehensive", "executive"], value="comprehensive",
                                           label="Summary type", elem_classes="input-field")
                    sum_btn = gr.Button("Generate summary", elem_classes="primary-button")
                with gr.Column(scale=2):
                    sum_out = gr.Markdown(elem_classes="results")
            sum_btn.click(generate_summary, inputs=[sum_doc_id, sum_type, st_api_url], outputs=[sum_out])

        # Settings
        with gr.Tab("Settings"):
            gr.Markdown("### ⚙️ API & Theme", elem_classes="panel")
            with gr.Row():
                api_box = gr.Textbox(value=DEFAULT_API_URL, label="Backend API base URL", elem_classes="input-field")
                set_btn = gr.Button("Save API URL", elem_classes="primary-button")
            set_msg = gr.Markdown(elem_classes="results")
            set_btn.click(set_api_url, inputs=[api_box], outputs=[st_api_url, set_msg])

            gr.Markdown("### 🎨 Theme", elem_classes="panel")
            theme_choice = gr.Radio(
                choices=["Dim (Neutral)", "Light (High Contrast)", "Dark (High Contrast)"],
                value="Dim (Neutral)",
                label="Choose theme",
                elem_classes="input-field",
            )
            theme_note = gr.Markdown("Theme applied.", elem_classes="results")
            theme_choice.change(
                lambda t: apply_theme(t),
                inputs=[theme_choice],
                outputs=[theme_slot, theme_note],
            )

    gr.HTML("""
    <div class="footer">
      <div>FastAPI · PostgreSQL + pgvector · Hybrid retrieval · Together.ai • © 2025 Demo</div>
      <div>Use via API · Built with Gradio</div>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
