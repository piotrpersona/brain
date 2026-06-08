import asyncio
import os
import json
import base64
from pathlib import Path
import httpx
from lightrag import LightRAG, QueryParam
from rlm import RLM # Upstream reference package: rlms (imported as rlm)

# ---------------------------------------------------------
# Storage Directories Configuration
# ---------------------------------------------------------
DATA_DIR = Path('./data')
RAG_WORKDIR = DATA_DIR / 'rag_working_dir'
VAULT_CONVERSATIONS = DATA_DIR / 'vault' / 'conversations'
MINERU_OUTPUT_DIR = DATA_DIR / 'extracted_content'

os.makedirs(RAG_WORKDIR, exist_ok=True)
os.makedirs(VAULT_CONVERSATIONS, exist_ok=True)
os.makedirs(MINERU_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Define Local Model Function Calls via Ollama Backend
# ---------------------------------------------------------

async def local_llm_model_func(prompt, system_prompt=None, history=[], model_id="qwen3:14b", **kwargs) -> str:
    """Handles text generation and graph entity extraction via Ollama."""
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
                "model": model_id,  # Primary text and coding model
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": 32768}
            }
        )
        return response.json()["message"]["content"]

async def local_vision_model_func(messages, model_id="qwen3-vl", **kwargs) -> str:
    """Passes multimodal context (text + base64 images) to Qwen2.5-VL."""
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_id,  # Local VLM for handling charts/tables/math layout elements
                "messages": messages,
                "stream": False
            }
        )
        return response.json()["message"]["content"]

# ---------------------------------------------------------
# 2. Initialize RAG-Anything Engine
# ---------------------------------------------------------
rag = LightRAG(
    working_dir=RAG_WORKDIR,
    llm_model_func=local_llm_model_func,
    vision_model_func=local_vision_model_func,
    addon_params={
        "embedding_model": "BAAI/bge-m3",
        "chunk_size": 1200,
        "chunk_overlap": 200
    }
)

# ---------------------------------------------------------
# 3. Model Context Protocol (MCP) Integration Engine
# ---------------------------------------------------------
async def search_arxiv_mcp(query: str, max_results: int = 5) -> str:
    """
    Spawns the globally installed arxiv-mcp-server process via standard I/O (stdio).
    Communicates using JSON-RPC 2.0 primitives compliant with the MCP spec.
    """
    cmd = ["uv", "tool", "run", "arxiv-mcp-server"]
    
    # Construct standard MCP tools/call payload
    mcp_payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "search_papers",
            "arguments": {
                "query": query,
                "max_results": max_results
            }
        },
        "id": 1
    }
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Write payload to process stdin and wait for response
        stdout, stderr = await process.communicate(input=json.dumps(mcp_payload).encode('utf-8'))
        
        if process.returncode != 0:
            return f"MCP Server Error: {stderr.decode('utf-8', errors='ignore')}"
            
        response_data = json.loads(stdout.decode('utf-8'))
        
        # Unpack standard MCP text content response array
        if "result" in response_data and "content" in response_data["result"]:
            contents = response_data["result"]["content"]
            return "\n".join([c.get("text", "") for c in contents if c.get("type") == "text"])
            
        return f"Unexpected MCP Server Response format: {str(response_data)}"
        
    except Exception as e:
        return f"Failed to execute ArXiv MCP Tool connection: {str(e)}"

# ---------------------------------------------------------
# 4. Memory Preservation Layer (Dumping Dialogues to Disk)
# ---------------------------------------------------------
def log_conversation_to_disk(session_id: str, role: str, content: str):
    """Saves dialog steps chronologically as JSONL files for the RLM loop to inspect."""
    log_file = VAULT_CONVERSATIONS / f"{session_id}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"role": role, "content": content}) + "\n")

# ---------------------------------------------------------
# 5. Pipeline Execution: MinerU -> RAG-Anything -> RLMS
# ---------------------------------------------------------

async def main():
    target_pdf = "sample_quantum_paper.pdf"
    session_id = "rag_architecture_session"
    
    # --- PHASE 1: MINERU DEEP PARSING ---
    print(f"[*] Executing MinerU layout extraction on: {target_pdf}")
    os.system(f"magic-pdf -i {target_pdf} -o {MINERU_OUTPUT_DIR} -m method_auto")
    
    # --- PHASE 2: RAG-ANYTHING ADOPTION ---
    print("\n[*] Ingesting MinerU extractions into RAG-Anything Graph backend...")
    await rag.process_document_complete(
        file_path=str(MINERU_OUTPUT_DIR),
        parse_method="auto", 
        device="cuda",       
        lang="en"
    )
    print("[✓] Knowledge Graph successfully synced.")

    # --- PHASE 3: INTERACTIVE USER QUERY & RECURSIVE SYNTHESIS ---
    user_query = "Check our local architecture history, run an external search on recent alternative RAG approaches via ArXiv, and compare them."
    
    log_conversation_to_disk(session_id, "user", user_query)
    
    print(f"\n[*] Initializing Recursive Language Model (RLM) workspace context loop...")
    rlm_orchestrator = RLM(
        backend="ollama",
        backend_kwargs={
            "model_name": "qwen3:14b", # Updated to Qwen 3 for enhanced dual-mode reasoning
            "base_url": "http://localhost:11434"
        },
        environment="local" 
    )

    # Expose helper tools directly into the RLM Python REPL workspace execution scope
    def r_query(q: str) -> str:
        """LightRAG Graph Engine search hook accessible via code inside RLM sandbox."""
        return asyncio.run(rag.aquery(q, mode="hybrid", vlm_enhanced=True))

    def arxiv_search(q: str) -> str:
        """Live external ArXiv MCP microservice call accessible via code inside RLM sandbox."""
        return asyncio.run(search_arxiv_mcp(q, max_results=5))

    rlm_prompt = f"""
    You are an AI research assistant orchestrating a local data vault.
    You have a local file directory containing past chat timelines here: '{VAULT_CONVERSATIONS}'
    
    You have programmatic access to the following built-in python tools inside your sandbox:
    - `r_query(string)`: Query the local Knowledge Graph for ingested paper structures and layouts.
    - `arxiv_search(string)`: Run a live external JSON-RPC search against the ArXiv database via MCP.
    
    User Request: {user_query}
    
    Task: Write and execute a python routine to look through past logs on disk, execute `r_query()` and `arxiv_search()` 
    where gaps exist, correlate the findings, and generate a final synthesized answer. Use `sub_llm_call()` if processing large chunks.
    """

    # Compute execution turn over disk parameters using RLM scaling
    rlm_execution = rlm_orchestrator.completion(task=rlm_prompt)
    final_output = rlm_execution.response
    
    print(f"\n[✓] System Response:\n{final_output}")
    
    log_conversation_to_disk(session_id, "assistant", final_output)

if __name__ == "__main__":
    asyncio.run(main())