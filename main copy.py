import asyncio
import os
import json
from pathlib import Path
import httpx
from lightrag import LightRAG
from lightrag.base import EmbeddingFunc

# Textual UI Imports
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static

# ---------------------------------------------------------
# Storage & Directory Architecture
# ---------------------------------------------------------
DATA_DIR = Path('./data')
RAG_WORKDIR = DATA_DIR / 'rag_working_dir'
VAULT_CONVERSATIONS = DATA_DIR / 'vault' / 'conversations'

os.makedirs(RAG_WORKDIR, exist_ok=True)
os.makedirs(VAULT_CONVERSATIONS, exist_ok=True)

# ---------------------------------------------------------
# Core Engine Initializations (LightRAG & MCP)
# ---------------------------------------------------------
async def local_llm_model_func(prompt, system_prompt=None, history=[], model_id="qwen3-vl", **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for hist in history:
        messages.append(hist)
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_id,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": 32768, "use_mlock": True}
            }
        )
        return response.json()["message"]["content"]

async def local_embedding_func(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text", 
                "input": texts
            }
        )
        return response.json()["embeddings"]

rag = LightRAG(
    working_dir=RAG_WORKDIR,
    llm_model_func=local_llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=768, 
        max_token_size=8192,
        func=local_embedding_func
    ),
    addon_params={"embedding_model": "BAAI/bge-m3", "chunk_size": 1200}
)

async def search_arxiv_mcp(query: str, max_results: int = 3) -> str:
    cmd = ["uv", "tool", "run", "arxiv-mcp-server"]
    mcp_payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "search_papers", "arguments": {"query": query, "max_results": max_results}},
        "id": 1
    }
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate(input=json.dumps(mcp_payload).encode('utf-8'))
        response_data = json.loads(stdout.decode('utf-8'))
        if "result" in response_data and "content" in response_data["result"]:
            return "\n".join([c.get("text", "") for c in response_data["result"]["content"] if c.get("type") == "text"])
        return "No papers found."
    except Exception as e:
        return f"MCP Error: {str(e)}"

# ---------------------------------------------------------
# Textual TUI Application Definition
# ---------------------------------------------------------
class AcademicAgentApp(App):
    """A clean, split-screen terminal environment for your agent."""
    
    CSS = """
    Screen {
        background: #1a1a1a;
    }
    #chat_pane {
        width: 65%;
        border-right: solid #333333;
        padding: 1;
    }
    #system_pane {
        width: 35%;
        padding: 1;
        background: #111111;
    }
    Input {
        dock: bottom;
        margin-top: 1;
        border: tall #444444;
    }
    Input:focus {
        border: tall #00ff00;
    }
    .pane-title {
        background: #222222;
        color: #00ff00;
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS = [("q", "quit", "Quit Application")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="chat_pane"):
                yield Static("💬 RESEARCH CONVERSATION", classes="pane-title")
                yield RichLog(id="chat_log", wrap=True, highlight=True)
                yield Input(placeholder="Ask about a research topic (or 'q' to exit)...")
            
            with Vertical(id="system_pane"):
                yield Static("⚙️ SYSTEM SUB-PROCESS LOG", classes="pane-title")
                yield RichLog(id="sys_log", wrap=True, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat_log").write(
            Text.from_markup("[bold green]System initialized.[/bold green] Enter your target topic below.")
        )
        self.query_one("#sys_log").write(
            Text.from_markup("[grey50]Awaiting sub-tasks...[/grey50]")
        )
        self.session_id = "tui_session_001"
        
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return
            
        if user_text.lower() == 'q':
            self.exit()
            return

        input_widget = event.input
        input_widget.value = ""
        chat_log = self.query_one("#chat_log")
        sys_log = self.query_one("#sys_log")
        
        chat_log.write(Text.from_markup(f"\n[bold cyan]User:[/bold cyan] {user_text}"))
        sys_log.write(Text.from_markup("[~] Spawning context-gathering tasks..."))

        # Handled natively as an async worker via Textual's async scheduler
        self.run_worker(self.execute_agent_loop(user_text, chat_log, sys_log))

    async def execute_agent_loop(self, query: str, chat_log: RichLog, sys_log: RichLog):
        """Pure asynchronous context gathering and direct local inference execution."""
        with open(VAULT_CONVERSATIONS / f"{self.session_id}.jsonl", "a") as f:
            f.write(json.dumps({"role": "user", "content": query}) + "\n")

        fetched_container = {"data": ""}

        try:
            # 1. Direct Graph-RAG lookup
            sys_log.write(f"[Graph-RAG] Scanning local entity indices for: '{query}'")
            local_context = await rag.aquery(query)
            
            # 2. Adaptive external discovery logic
            mcp_context = ""
            if len(str(local_context).strip()) < 150 or "arxiv" in query.lower() or "fresh" in query.lower():
                sys_log.write(f"[MCP] Local bounds minimal or external refresh forced. Polling ArXiv...")
                mcp_context = await search_arxiv_mcp(query)
                fetched_container["data"] = mcp_context
                sys_log.write(f"[MCP] Harvested context parameters.")
            else:
                sys_log.write(f"[System] Retained functional baseline metrics from RAG indices.")

            # 3. Formulate prompt context injection payload
            unified_prompt = f"""You are an elite research assistant. Synthesize a comprehensive, authoritative response based on the provided local knowledge graph extracts and external literature.

[LOCAL KNOWLEDGE GRAPH CONTEXT]
{local_context}

[EXTERNAL LITERATURE CONTEXT (MCP)]
{mcp_context if mcp_context else 'No external lookup performed.'}

USER QUERY: {query}

Provide a rigorous synthesis below. If context sources conflict, note the discrepancy.
"""

            # 4. Straight evaluation step against Ollama
            sys_log.write("[Ollama] Processing local inference pipeline...")
            response = await local_llm_model_func(prompt=unified_prompt, model_id="qwen3-vl")
            
            # 5. Flush directly back to UI thread
            chat_log.write(f"\n[bold green]Assistant:[/bold green] {response}")
            
            with open(VAULT_CONVERSATIONS / f"{self.session_id}.jsonl", "a") as f:
                f.write(json.dumps({"role": "assistant", "content": response}) + "\n")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            chat_log.write(f"\n[bold red]Pipeline Error Encountered:[/bold red] {str(e)}")
            sys_log.write(f"[ERROR DETAILS]:\n{error_details}")

        # 6. Smooth Background Graph Assimilation
        if fetched_container["data"]:
            sys_log.write("[Background Task] Commencing Graph-RAG matrix update...")
            try:
                await rag.ainsert(fetched_container["data"])
                sys_log.write("[Background Task] ✓ Knowledge Graph updated successfully.")
            except Exception as e:
                sys_log.write(f"[Background Task] ✗ Ingestion failed: {str(e)}")

if __name__ == "__main__":
    app = AcademicAgentApp()
    app.run()