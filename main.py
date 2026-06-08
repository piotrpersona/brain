import asyncio
import os
import json
from pathlib import Path
import httpx
from lightrag import LightRAG
from rlm import RLM

# Storage Setup
DATA_DIR = Path('./data')
RAG_WORKDIR = DATA_DIR / 'rag_working_dir'
VAULT_CONVERSATIONS = DATA_DIR / 'vault' / 'conversations'

os.makedirs(RAG_WORKDIR, exist_ok=True)
os.makedirs(VAULT_CONVERSATIONS, exist_ok=True)

# ---------------------------------------------------------
# Ollama and LightRAG Base Initializations
# ---------------------------------------------------------
async def local_llm_model_func(prompt, system_prompt=None, history=[], model_id="qwen3:14b", **kwargs) -> str:
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

rag = LightRAG(
    working_dir=RAG_WORKDIR,
    llm_model_func=local_llm_model_func,
    addon_params={"embedding_model": "BAAI/bge-m3", "chunk_size": 1200}
)

# ---------------------------------------------------------
# ArXiv MCP Client Communication Bridge
# ---------------------------------------------------------
async def search_arxiv_mcp(query: str, max_results: int = 3) -> str:
    """Invokes the global arxiv-mcp-server process via standard I/O."""
    cmd = ["uv", "tool", "run", "arxiv-mcp-server"]
    mcp_payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "search_papers",
            "arguments": {"query": query, "max_results": max_results}
        },
        "id": 1
    }
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate(input=json.dumps(mcp_payload).encode('utf-8'))
        response_data = json.loads(stdout.decode('utf-8'))
        if "result" in response_data and "content" in response_data["result"]:
            contents = response_data["result"]["content"]
            return "\n".join([c.get("text", "") for c in contents if c.get("type") == "text"])
        return ""
    except Exception as e:
        return f"ArXiv Search Error: {str(e)}"

# ---------------------------------------------------------
# Dynamic Background Indexing Loop
# ---------------------------------------------------------
async def background_index_worker(text_content: str):
    """
    Fires off silently in the background. Parses and splits incoming text 
    into entities/relationships and merges them into the permanent Graph-RAG.
    """
    if not text_content or "Error" in text_content:
        return
    print("\n[Background Task] Starting text-chunking and knowledge graph insertion...")
    try:
        # Use LightRAG's native ainsert to parse raw strings directly into your graph vectors
        await rag.ainsert(text_content)
        print("[Background Task] ✓ Graph database updated with fresh paper entities.")
    except Exception as e:
        print(f"[Background Task] ✗ Indexing failed: {str(e)}")

# ---------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------
async def main():
    # 1. Simulate a brand new chat session run
    session_id = "session_run_001" 
    user_query = "What are the latest breakthroughs in Kolmogorov-Arnold Networks (KAN) architectures?"
    
    print(f"[*] Initializing Session: {session_id}")
    print(f"[*] User Request: {user_query}")

    # 2. Setup the sandboxed workspace environment
    rlm_orchestrator = RLM(
        backend="ollama",
        backend_kwargs={"model_name": "qwen3:14b", "base_url": "http://localhost:11434"},
        environment="local" 
    )

    # Global tracking container to capture the fetched papers out of the RLM loop execution
    fetched_data_container = {"payload": ""}

    # Define tools exposed directly into the RLM runtime environment
    def r_query(q: str) -> str:
        return asyncio.run(rag.aquery(q, mode="hybrid"))

    def arxiv_search(q: str) -> str:
        raw_papers = asyncio.run(search_arxiv_mcp(q))
        # Save results to container so the master script can process it in the background
        fetched_data_container["payload"] += f"\n\n--- Source: {q} ---\n{raw_papers}"
        return raw_papers

    rlm_prompt = f"""
    You are an architecture research assistant. You have access to:
    - `r_query(string)`: Look for concepts already saved in our local knowledge graph.
    - `arxiv_search(string)`: Fetch real-time paper summaries directly from ArXiv via MCP.
    
    User Query: {user_query}
    
    Task: Check if we have information locally via `r_query`. If nothing matches or more context is needed, 
    call `arxiv_search` to pull fresh papers, compile a great response, and output it.
    """

    # 3. Compute response interactively
    rlm_execution = rlm_orchestrator.completion(task=rlm_prompt)
    print(f"\n[✓] System Response:\n{rlm_execution.response}")

    # 4. Fire-and-Forget Ingestion
    # The user gets their answer instantly while the indexing runs concurrently in the background
    if fetched_data_container["payload"]:
        asyncio.create_task(background_index_worker(fetched_data_container["payload"]))
        
    # Keep the loop alive briefly to let the background tasks finish initialization steps
    await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())