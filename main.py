import asyncio
import os
import json
from pathlib import Path
import httpx
from lightrag import LightRAG
from rlm import RLM
from lightrag.base import EmbeddingFunc

# Textual UI Imports
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static
from textual.worker import get_current_worker

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

# 1. Define the Ollama embedding function
async def local_embedding_func(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text", # Or "bge-m3" if you have it in Ollama
                "input": texts
            }
        )
        # Ollama returns a dict with an "embeddings" key containing arrays of floats
        return response.json()["embeddings"]

rag = LightRAG(
    working_dir=RAG_WORKDIR,
    llm_model_func=local_llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=768, # nomic-embed-text uses 768. (bge-m3 uses 1024)
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
            # Left Column: Interactive Chat Interface
            with Vertical(id="chat_pane"):
                yield Static("💬 RESEARCH CONVERSATION", classes="pane-title")
                yield RichLog(id="chat_log", wrap=True, highlight=True)
                yield Input(placeholder="Ask about a research topic (or 'q' to exit)...")
            
            # Right Column: Real-time System Tasks and Ingestion Status
            with Vertical(id="system_pane"):
                yield Static("⚙️ SYSTEM SUB-PROCESS LOG", classes="pane-title")
                yield RichLog(id="sys_log", wrap=True, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        """Initialize UI text elements when app loads."""
        # Add markup=True here
        self.query_one("#chat_log").write(
            Text.from_markup("[bold green]System initialized.[/bold green] Enter your target topic below.")
        )
        self.query_one("#sys_log").write(
            Text.from_markup("[grey50]Awaiting sub-tasks...[/grey50]")
        )
        self.session_id = "tui_session_001"
        
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Fires whenever you hit enter in the input field."""
        user_text = event.value.strip()
        if not user_text:
            return
            
        if user_text.lower() == 'q':
            self.exit()
            return

        # Clear input field and print user message to chat log
        input_widget = event.input
        input_widget.value = ""
        chat_log = self.query_one("#chat_log")
        sys_log = self.query_one("#sys_log")
        
        chat_log.write(Text.from_markup(f"\n[bold cyan]User:[/bold cyan] {user_text}"))
        sys_log.write(Text.from_markup("[~] Launching RLM sandbox thread for query..."))

        self.run_worker(self.execute_agent_loop(user_text, chat_log, sys_log), thread=True)

    async def execute_agent_loop(self, query: str, chat_log: RichLog, sys_log: RichLog):
        """Asynchronous execution channel handling loop computations."""
        # File Logging
        with open(VAULT_CONVERSATIONS / f"{self.session_id}.jsonl", "a") as f:
            f.write(json.dumps({"role": "user", "content": query}) + "\n")

        rlm_orchestrator = RLM(
            backend="gemini",
            backend_kwargs={"model_name": "gemini-3.1-flash-lite"},
            # backend_kwargs={"model_name": "qwen3-vl", "base_url": "http://localhost:11434"},
            environment="local"
        )

        fetched_container = {"data": ""}

        # Injection Hooks that explicitly log back out to the UI panes
        def r_query(q: str) -> str:
            sys_log.write(f"[Graph-RAG] Scanning local entity indices for: '{q}'")
            return asyncio.run(rag.aquery(q, mode="hybrid"))

        def arxiv_search(q: str) -> str:
            sys_log.write(f"[MCP] Accessing ArXiv microservice for: '{q}'")
            papers = asyncio.run(search_arxiv_mcp(q))
            fetched_container["data"] += f"\n\n{papers}"
            sys_log.write(f"[MCP] Harvested new assets.")
            return papers

        rlm_prompt = f"""
        You are an elite research assistant with code-execution workspace privileges.
        Tools: `r_query(str)`, `arxiv_search(str)`. User Query: {query}
        Check local context first. Trigger MCP if fresh data is missing. Return a thorough synthesis.
        """

        # Compute completion step
        execution = rlm_orchestrator.completion(prompt=rlm_prompt)
        
        # Display output directly to UI chat thread
        chat_log.write(f"\n[bold green]Assistant:[/bold green] {execution.response}")
        
        with open(VAULT_CONVERSATIONS / f"{self.session_id}.jsonl", "a") as f:
            f.write(json.dumps({"role": "assistant", "content": execution.response}) + "\n")

        # Handle Background Assimilation Thread seamlessly
        if fetched_container["data"]:
            sys_log.write("[Background Task] Commencing Graph-RAG ingestion...")
            try:
                await rag.ainsert(fetched_container["data"])
                sys_log.write("[Background Task] ✓ Knowledge Graph updated successfully.")
            except Exception as e:
                sys_log.write(f"[Background Task] ✗ Ingestion failed: {str(e)}")

# Run the TUI
if __name__ == "__main__":
    app = AcademicAgentApp()
    app.run()