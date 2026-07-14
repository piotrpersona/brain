#!/usr/bin/env python3
import asyncio
import os
import uuid
from typing import Dict, List, Optional
import tempfile
import httpx
from dotenv import load_dotenv

load_dotenv()

# Official Google GenAI SDK
from google import genai
from google.genai import types

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# Unified Configuration Variables for Gemini
# Note: GEMINI_API_KEY will be automatically picked up by genai.Client() if set in env
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2072")) # Gemini default is 768

# Initialize the Gemini Client
client = genai.Client()

async def gemini_llm_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    """Native Gemini LLM implementation mapping OpenAI-style history formats."""
    # Convert system prompt to Gemini's GenerateContentConfig format if present
    config = types.GenerateContentConfig()
    if system_prompt:
        config.system_instruction = system_prompt
        
    # Map extra parameters (like temperature) if passed in kwargs
    if "temperature" in kwargs:
        config.temperature = kwargs["temperature"]

    # If there's an existing chat history, we map it to Gemini's types.Content structure
    if history_messages:
        contents = []
        for msg in history_messages:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.get("content", ""))]
                )
            )
        # Append the final prompt as the latest turn
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        )
        
        # We use standard client.aio for async API calls
        response = await client.aio.models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config=config
        )
    else:
        # Simple standalone prompt execution
        response = await client.aio.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=config
        )
        
    return response.text


async def gemini_embedding_async(texts: List[str]) -> List[List[float]]:
    """Native Gemini async embedding function."""
    response = await client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        # Gemini allows configuring specific dimensions if needed via config:
        # config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
    )
    # Extract out the floating point lists from the response object
    return [embedding.values for embedding in response.embeddings]


class GeminiRAGIntegration:
    """Integration class for RAG-Anything powered by Gemini."""

    def __init__(self):
        self.llm_model = LLM_MODEL
        self.embedding_model = EMBEDDING_MODEL
        self.embedding_dim = EMBEDDING_DIM

        self.config = RAGAnythingConfig(
            working_dir=f"./data/rag_storage_gemini/{uuid.uuid4()}",
            parser="docling",
            parse_method="auto",
            enable_image_processing=False,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        print(f"📁 Using working_dir: {self.config.working_dir}")
        self.rag = None

    async def test_embedding(self) -> bool:
        """Sanity-check for the Gemini embedding endpoint."""
        try:
            print(f"🔢 Testing embedding model: {self.embedding_model}")
            vectors = await gemini_embedding_async(["hello world"])
            if vectors and len(vectors[0]) > 0:
                actual_dim = len(vectors[0])
                print(f"✅ Embedding OK — dim={actual_dim} (configured: {self.embedding_dim})")
                if actual_dim != self.embedding_dim:
                    print(f"  ⚠️ Dimension mismatch! Got {actual_dim}, configured {self.embedding_dim}")
                return True
            print("❌ Embedding returned empty vector")
            return False
        except Exception as e:
            print(f"❌ Embedding test failed: {e}")
            return False

    async def test_chat(self) -> bool:
        """Sanity-check for the Gemini LLM endpoint."""
        try:
            print(f"💬 Testing LLM model: {self.llm_model}")
            result = await gemini_llm_model_func("Say 'OK' in one word.")
            print(f"✅ Chat OK — response: {result.strip()[:80]}")
            return True
        except Exception as e:
            print(f"❌ Chat test failed: {e}")
            return False

    def _make_embedding_func(self) -> EmbeddingFunc:
        return EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=2048, # Gemini embeddings support up to 2048 tokens per segment
            func=gemini_embedding_async,
        )

    async def initialize_rag(self) -> bool:
        """Initialize RAG-Anything with Gemini endpoints."""
        print("\nInitializing RAG-Anything with Gemini...")
        try:
            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=gemini_llm_model_func,
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
            async with httpx.AsyncClient() as client_http:
                response = await client_http.get(document_uri)
                document_content = response.content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(document_content)
                document_path = temp_pdf.name
        try:
            print(f"📄 Processing: {document_path}")
            await self.rag.process_document_complete(
                file_path=document_path,
                output_dir="./output_gemini",
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
    print("Google Gemini API + RAG-Anything Integration")
    print("=" * 70)

    # Ensure your GEMINI_API_KEY is available in your environment before executing
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        return False

    integration = GeminiRAGIntegration()

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