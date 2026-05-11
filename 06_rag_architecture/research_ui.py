"""
Interactive Web UI for exploring research_assistant.py.
Run:  python 06_rag_architecture/research_ui.py
Then open: http://localhost:8000
"""

import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from research_assistant import AIResearchAssistant, DB_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# App + singleton
# ---------------------------------------------------------------------------

app = FastAPI(title="Research Assistant UI")
assistant: AIResearchAssistant = AIResearchAssistant()
server_logs: list[str] = []


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    server_logs.append(entry)
    print(entry)


# ---------------------------------------------------------------------------
# Demo documents (verbatim from run_research_assistant_demo)
# ---------------------------------------------------------------------------

DEMO_DOCS = [
    (
        """
        Attention Mechanisms in Neural Networks

        The attention mechanism was introduced in "Attention Is All You Need"
        by Vaswani et al. (2017). It allows models to focus on relevant parts
        of the input when generating output.

        Key concepts:
        - Query, Key, Value (QKV) triplets
        - Scaled dot-product attention
        - Multi-head attention for parallel processing

        The transformer architecture has become the foundation for modern NLP
        models including BERT, GPT, and T5.
        """,
        "attention_mechanisms.pdf",
    ),
    (
        """
        Retrieval-Augmented Generation (RAG)

        RAG combines retrieval systems with generative models. First introduced
        by Lewis et al. (2020), RAG addresses the limitation of LLMs being
        limited to their training data.

        Components of a RAG system:
        1. Document store with vector embeddings
        2. Retriever to find relevant documents
        3. Generator (LLM) to produce responses

        Benefits include reduced hallucination, up-to-date information,
        and source attribution.
        """,
        "rag_survey.pdf",
    ),
    (
        """
        LangChain and LangGraph Framework Overview

        LangChain is an open-source framework for building LLM applications.
        Key features include modular components, integration with 50+ LLM
        providers, and built-in RAG utilities.

        LangGraph extends LangChain for stateful applications with
        graph-based state management, support for cycles and loops,
        and human-in-the-loop workflows.
        """,
        "langchain_docs.md",
    ),
]

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AddTextRequest(BaseModel):
    source: str
    text: str


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    use_advanced: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/status")
def status():
    return {
        "chunk_count": assistant.get_document_count(),
        "sources": assistant.list_sources(),
        "logs": list(server_logs),
    }


@app.post("/load-demo")
def load_demo():
    for text, source in DEMO_DOCS:
        assistant.add_text(text, source=source)
    count = assistant.get_document_count()
    log(f"Loaded demo documents → {count} chunks indexed")
    return {"chunk_count": count, "sources": assistant.list_sources()}


@app.post("/add-text")
def add_text(req: AddTextRequest):
    if not req.source.strip() or not req.text.strip():
        raise HTTPException(status_code=400, detail="source and text are required")
    chunks = assistant.add_text(req.text, source=req.source)
    log(f"Added text from '{req.source}' → {chunks} new chunks")
    return {"chunks_added": chunks, "total": assistant.get_document_count()}


@app.post("/ask")
def ask(req: AskRequest):
    if assistant.get_document_count() == 0:
        raise HTTPException(status_code=400, detail="No documents loaded. Use /load-demo or /add-text first.")
    log(f"ASK (advanced={req.use_advanced}, session={req.session_id!r}): {req.question[:80]}")
    response = assistant.ask_structured(req.question, req.session_id, req.use_advanced)
    log(f"Response received — confidence: {response.confidence}")
    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "sources": response.sources,
        "key_quotes": response.key_quotes,
        "follow_up_questions": response.follow_up_questions,
        "logs": list(server_logs),
    }


@app.get("/session/{session_id}/history")
def session_history(session_id: str):
    return {"history": assistant.get_session_messages(session_id)}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    assistant.clear_session(session_id)
    log(f"Cleared session '{session_id}'")
    return {"ok": True}


@app.delete("/reset")
def reset():
    global assistant
    assistant.session_store.clear()
    shutil.rmtree(DB_DIR, ignore_errors=True)
    assistant = AIResearchAssistant()
    server_logs.clear()
    log("Reset: database and sessions cleared")
    return {"ok": True}


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Assistant Explorer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  h1 { font-size: 1.25rem; font-weight: 700; }
  h2 { font-size: .85rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: #94a3b8; margin-bottom: .75rem; }
  header { display: flex; align-items: center; justify-content: space-between; padding: .9rem 1.25rem; background: #1e2430; border-bottom: 1px solid #2d3748; }
  .layout { display: grid; grid-template-columns: 260px 1fr; grid-template-rows: 1fr auto auto; height: calc(100vh - 49px); }
  /* Sidebar */
  aside { grid-row: 1 / 4; background: #161b27; border-right: 1px solid #2d3748; padding: 1rem; display: flex; flex-direction: column; gap: 1rem; overflow-y: auto; }
  /* Chat */
  .chat-area { padding: 1rem; overflow-y: auto; display: flex; flex-direction: column; gap: .5rem; }
  .msg { padding: .6rem .85rem; border-radius: 8px; max-width: 88%; line-height: 1.5; font-size: .9rem; }
  .msg.user { background: #2563eb; align-self: flex-end; }
  .msg.ai { background: #1e2430; border: 1px solid #2d3748; align-self: flex-start; }
  /* Input bar */
  .input-bar { display: flex; gap: .5rem; padding: .75rem 1rem; background: #1e2430; border-top: 1px solid #2d3748; }
  .input-bar input { flex: 1; background: #0f1117; border: 1px solid #374151; color: #e2e8f0; border-radius: 6px; padding: .5rem .75rem; font-size: .9rem; }
  .input-bar input:focus { outline: none; border-color: #3b82f6; }
  /* Response panel */
  .response-panel { background: #1a2035; border-top: 1px solid #2d3748; padding: 1rem; overflow-y: auto; max-height: 280px; }
  .response-panel.hidden { display: none; }
  .confidence { display: inline-block; padding: .2rem .6rem; border-radius: 99px; font-size: .75rem; font-weight: 700; text-transform: uppercase; }
  .conf-high { background: #065f46; color: #6ee7b7; }
  .conf-medium { background: #78350f; color: #fcd34d; }
  .conf-low { background: #7f1d1d; color: #fca5a5; }
  .sources-row { display: flex; flex-wrap: wrap; gap: .35rem; margin: .5rem 0; }
  .source-tag { background: #1e3a5f; color: #93c5fd; padding: .15rem .5rem; border-radius: 4px; font-size: .75rem; }
  .section-label { font-size: .75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: .07em; margin: .75rem 0 .35rem; }
  .quote { border-left: 3px solid #3b82f6; padding-left: .6rem; color: #94a3b8; font-style: italic; font-size: .85rem; margin-bottom: .4rem; }
  .followup { color: #60a5fa; font-size: .85rem; cursor: pointer; padding: .2rem 0; }
  .followup:hover { text-decoration: underline; }
  /* Logs */
  .logs-panel { border-top: 1px solid #2d3748; background: #0c1018; }
  .logs-header { display: flex; align-items: center; justify-content: space-between; padding: .4rem .9rem; background: #161b27; font-size: .75rem; cursor: pointer; user-select: none; }
  .logs-body { font-family: monospace; font-size: .75rem; color: #64748b; padding: .5rem .9rem; max-height: 130px; overflow-y: auto; }
  .log-line { padding: .1rem 0; }
  /* Shared UI */
  button { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: .45rem .85rem; font-size: .85rem; cursor: pointer; }
  button:hover { background: #1d4ed8; }
  button.secondary { background: #1e2430; border: 1px solid #374151; color: #94a3b8; }
  button.secondary:hover { background: #2d3748; }
  button.danger { background: #7f1d1d; color: #fca5a5; border: 1px solid #991b1b; }
  button.danger:hover { background: #991b1b; }
  input[type=text], textarea { background: #0f1117; border: 1px solid #374151; color: #e2e8f0; border-radius: 6px; padding: .45rem .65rem; font-size: .85rem; width: 100%; }
  input[type=text]:focus, textarea:focus { outline: none; border-color: #3b82f6; }
  textarea { resize: vertical; min-height: 80px; font-family: inherit; }
  label { font-size: .8rem; color: #94a3b8; display: block; margin-bottom: .25rem; }
  .status-bar { font-size: .75rem; color: #64748b; padding: .35rem .9rem; background: #161b27; border-top: 1px solid #1e2430; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #3b82f6; border-top-color: transparent; border-radius: 50%; animation: spin .6s linear infinite; vertical-align: middle; margin-right: .4rem; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .session-row { display: flex; gap: .5rem; align-items: center; }
  .session-row input { flex: 1; }
  .checkbox-row { display: flex; align-items: center; gap: .5rem; font-size: .85rem; }
  .checkbox-row input[type=checkbox] { width: auto; accent-color: #3b82f6; }
</style>
</head>
<body>

<header>
  <h1>Research Assistant Explorer</h1>
  <button class="danger" onclick="resetAll()">Reset All</button>
</header>

<div class="layout">

  <!-- Sidebar -->
  <aside>
    <h2>Setup</h2>

    <button onclick="loadDemo()">Load Demo Documents</button>

    <div style="border-top:1px solid #2d3748;padding-top:.75rem">
      <h2>Add Custom Text</h2>
      <label>Source name</label>
      <input type="text" id="src" placeholder="my_paper.pdf">
      <label style="margin-top:.5rem">Text</label>
      <textarea id="txt" placeholder="Paste or type document text…"></textarea>
      <button style="margin-top:.5rem;width:100%" onclick="addText()">Add Text</button>
    </div>

    <div style="border-top:1px solid #2d3748;padding-top:.75rem">
      <h2>Session</h2>
      <div class="session-row">
        <input type="text" id="sessionId" value="default">
        <button class="secondary" onclick="clearSession()" title="Clear history">✕</button>
      </div>
      <div class="checkbox-row" style="margin-top:.5rem">
        <input type="checkbox" id="useAdvanced" checked>
        <label style="margin:0">Multi-query retrieval</label>
      </div>
    </div>
  </aside>

  <!-- Chat -->
  <div class="chat-area" id="chat"></div>

  <!-- Input bar -->
  <div class="input-bar">
    <input type="text" id="questionInput" placeholder="Ask a question about your documents…"
           onkeydown="if(event.key==='Enter')ask()">
    <button onclick="ask()" id="askBtn">Ask</button>
  </div>

  <!-- Structured response panel -->
  <div class="response-panel hidden" id="responsePanel">
    <div style="display:flex;align-items:center;gap:.75rem;flex-wrap:wrap">
      <h2 style="margin:0">Last Response</h2>
      <span class="confidence" id="confBadge"></span>
      <div class="sources-row" id="sourcesRow"></div>
    </div>
    <div class="section-label">Answer</div>
    <div id="answerText" style="font-size:.9rem;line-height:1.6"></div>
    <div class="section-label" id="quotesLabel">Key Quotes</div>
    <div id="quotesContainer"></div>
    <div class="section-label" id="followupLabel">Follow-up Questions</div>
    <div id="followupsContainer"></div>
  </div>

</div>

<!-- Status bar -->
<div class="status-bar" id="statusBar">Loading…</div>

<!-- Logs -->
<div class="logs-panel">
  <div class="logs-header" onclick="toggleLogs()">
    <span>Logs</span><span id="logsToggle">▼</span>
  </div>
  <div class="logs-body" id="logsBody"></div>
</div>

<script>
let logsVisible = true;

function toggleLogs() {
  logsVisible = !logsVisible;
  document.getElementById('logsBody').style.display = logsVisible ? 'block' : 'none';
  document.getElementById('logsToggle').textContent = logsVisible ? '▼' : '▶';
}

function renderLogs(logs) {
  const body = document.getElementById('logsBody');
  body.innerHTML = logs.map(l => `<div class="log-line">${esc(l)}</div>`).join('');
  body.scrollTop = body.scrollHeight;
}

async function refreshStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    const srcs = d.sources.length ? d.sources.join(', ') : 'none';
    document.getElementById('statusBar').textContent =
      `Chunks: ${d.chunk_count}  |  Sources: ${srcs}`;
    renderLogs(d.logs);
  } catch {}
}

async function loadDemo() {
  const r = await fetch('/load-demo', {method:'POST'});
  const d = await r.json();
  addLog(`Loaded demo → ${d.chunk_count} chunks`);
  refreshStatus();
}

async function addText() {
  const source = document.getElementById('src').value.trim();
  const text = document.getElementById('txt').value.trim();
  if (!source || !text) { alert('Fill in source and text.'); return; }
  const r = await fetch('/add-text', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({source, text})
  });
  const d = await r.json();
  addLog(`Added '${source}' → ${d.chunks_added} chunks`);
  document.getElementById('src').value = '';
  document.getElementById('txt').value = '';
  refreshStatus();
}

async function ask() {
  const q = document.getElementById('questionInput').value.trim();
  if (!q) return;
  const session = document.getElementById('sessionId').value.trim() || 'default';
  const useAdvanced = document.getElementById('useAdvanced').checked;

  document.getElementById('questionInput').value = '';
  appendMsg('user', q);

  const btn = document.getElementById('askBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Thinking…';

  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q, session_id: session, use_advanced: useAdvanced})
    });
    if (!r.ok) {
      const err = await r.json();
      appendMsg('ai', '⚠ ' + (err.detail || 'Error'));
      return;
    }
    const d = await r.json();
    appendMsg('ai', d.answer);
    renderStructured(d);
    renderLogs(d.logs);
    refreshStatus();
  } finally {
    btn.disabled = false;
    btn.textContent = 'Ask';
  }
}

function renderStructured(d) {
  const panel = document.getElementById('responsePanel');
  panel.classList.remove('hidden');

  const badge = document.getElementById('confBadge');
  badge.textContent = d.confidence;
  badge.className = 'confidence conf-' + (d.confidence || 'low');

  document.getElementById('sourcesRow').innerHTML =
    (d.sources || []).map(s => `<span class="source-tag">${esc(s)}</span>`).join('');

  document.getElementById('answerText').textContent = d.answer;

  const quotes = d.key_quotes || [];
  document.getElementById('quotesLabel').style.display = quotes.length ? '' : 'none';
  document.getElementById('quotesContainer').innerHTML =
    quotes.map(q => `<div class="quote">${esc(q)}</div>`).join('');

  const fqs = d.follow_up_questions || [];
  document.getElementById('followupLabel').style.display = fqs.length ? '' : 'none';
  document.getElementById('followupsContainer').innerHTML =
    fqs.map(fq => `<div class="followup" onclick="fillQuestion(this.textContent)">• ${esc(fq)}</div>`).join('');
}

function fillQuestion(text) {
  document.getElementById('questionInput').value = text.replace(/^[•]\s*/, '');
  document.getElementById('questionInput').focus();
}

async function clearSession() {
  const id = document.getElementById('sessionId').value.trim() || 'default';
  await fetch(`/session/${encodeURIComponent(id)}`, {method:'DELETE'});
  document.getElementById('chat').innerHTML = '';
  document.getElementById('responsePanel').classList.add('hidden');
  addLog(`Cleared session '${id}'`);
}

async function resetAll() {
  if (!confirm('Reset everything? This clears the database and all sessions.')) return;
  await fetch('/reset', {method:'DELETE'});
  document.getElementById('chat').innerHTML = '';
  document.getElementById('responsePanel').classList.add('hidden');
  refreshStatus();
}

function appendMsg(role, text) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addLog(msg) {
  const ts = new Date().toLocaleTimeString('en-GB');
  const body = document.getElementById('logsBody');
  const line = document.createElement('div');
  line.className = 'log-line';
  line.textContent = `[${ts}] ${msg}`;
  body.appendChild(line);
  body.scrollTop = body.scrollHeight;
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

refreshStatus();
setInterval(refreshStatus, 10000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("research_ui:app", host="0.0.0.0", port=8000, reload=True)
