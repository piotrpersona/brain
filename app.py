import asyncio
import os
import uuid
from typing import Dict, List, Optional
import tempfile
import httpx
from dotenv import load_dotenv

load_dotenv()

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache

# Unified Configuration Variables
MODEL_API_HOST = os.getenv("MODEL_API_HOST", "https://openrouter.ai/api/v1") 
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-14b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "your_api_key_here")

# Normalize base URL (strip trailing slashes, handle missing /v1 if needed)
MODEL_API_BASE_URL = MODEL_API_HOST.rstrip("/")


async def generic_llm_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    """Universal OpenAI-compatible LLM function."""
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=MODEL_API_BASE_URL,
        api_key=MODEL_API_KEY,
        **kwargs,
    )


async def generic_embedding_async(texts: List[str]) -> List[List[float]]:
    """Universal OpenAI-compatible embedding function using httpx."""
    headers = {
        "Authorization": f"Base {MODEL_API_KEY}" if "Bearer " in MODEL_API_KEY else f"Bearer {MODEL_API_KEY}",
        "Content-Type": "application/json",
    }
    # OpenRouter tracking headers (Optional)
    if "openrouter.ai" in MODEL_API_BASE_URL:
        headers["HTTP-Referer"] = "https://localhost:3000"
        headers["X-Title"] = "RAG-Anything"

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }
    
    async with httpx.AsyncClient() as client:
        # Construct path carefully depending on if /v1 is already in the host
        url = f"{MODEL_API_BASE_URL}/embeddings"
        response = await client.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


class GenericRAGIntegration:
    """Universal Integration class for RAG-Anything."""

    def __init__(self):
        self.llm_model = LLM_MODEL
        self.embedding_model = EMBEDDING_MODEL
        self.embedding_dim = EMBEDDING_DIM

        self.config = RAGAnythingConfig(
            working_dir=f"./data/rag_storage_generic/{uuid.uuid4()}",
            parser="docling",
            parse_method="auto",
            enable_image_processing=False,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        print(f"📁 Using working_dir: {self.config.working_dir}")
        self.rag = None

    async def test_embedding(self) -> bool:
        """Sanity-check for the embedding endpoint."""
        try:
            print(f"🔢 Testing embedding model: {self.embedding_model}")
            vectors = await generic_embedding_async(["hello world"])
            if vectors and len(vectors[0]) > 0:
                print(f"✅ Embedding OK — dim={len(vectors[0])} (configured: {self.embedding_dim})")
                if len(vectors[0]) != self.embedding_dim:
                    print(f"  ⚠️ Dimension mismatch! Got {len(vectors[0])}, configured {self.embedding_dim}")
                return True
            print("❌ Embedding returned empty vector")
            return False
        except Exception as e:
            print(f"❌ Embedding test failed: {e}")
            return False

    async def test_chat(self) -> bool:
        """Sanity-check for the LLM endpoint."""
        try:
            print(f"💬 Testing LLM model: {self.llm_model}")
            result = await generic_llm_model_func("Say 'OK' in one word.")
            print(f"✅ Chat OK — response: {result.strip()[:80]}")
            return True
        except Exception as e:
            print(f"❌ Chat test failed: {e}")
            return False

    def _make_embedding_func(self) -> EmbeddingFunc:
        return EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=8192,
            func=generic_embedding_async,
        )

    async def initialize_rag(self) -> bool:
        """Initialize RAG-Anything."""
        print("\nInitializing RAG-Anything ...")
        try:
            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=generic_llm_model_func,
                embedding_func=self._make_embedding_func(),
            )
            print("✅ RAG-Anything initialized")
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    async def process_document(self, document_uri: str):
        if not self.rag:
            print("❌ Call initialize_rag() first")
            return
        
        document_path = document_uri
        if document_uri.startswith("http"):
            print(f"⬇️ Fetching document: {document_path[:120]}")
            async with httpx.AsyncClient() as client:
                response = await client.get(document_uri)
                document_content = response.content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(document_content)
                document_path = temp_pdf.name
        try:
            print(f"📄 Processing: {document_path}")
            await self.rag.process_document_complete(
                file_path=document_path,
                output_dir="./output_generic",
                parse_method="auto",
                display_stats=True,
            )
            print("✅ Processing complete")
        finally:
            if os.path.exists(document_path) and document_uri.startswith("http"):
                os.remove(document_path)

    async def simple_query_example(self):
        if not self.rag:
            return
        result = await self.rag.aquery(
            "What is the ModernBERT?",
            mode="hybrid",
        )
        print(f"Answer: {result[:400]}")


async def main():
    print("=" * 70)
    print("Generic API + RAG-Anything Integration")
    print("=" * 70)

    integration = GenericRAGIntegration()

    if not await integration.test_embedding():
        return False

    if not await integration.test_chat():
        return False

    if not await integration.initialize_rag():
        return False

    await integration.process_document("https://arxiv.org/pdf/2412.13663")
    await integration.simple_query_example()
    return True


if __name__ == "__main__":
    asyncio.run(main())