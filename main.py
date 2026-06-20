import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_arxiv_mcp() -> str:
    server_params = StdioServerParameters(
        command="uv",
        args=["tool", "run", "arxiv-mcp-server"]
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                test_query = "ModernBERT" 
                print(f"Sending query: {test_query}...")

                result = await session.call_tool(
                    "search_papers", 
                    arguments={"query": test_query, "max_results": 1}
                )

                # DEBUG: Print exactly what the MCP server gives us back
                print("\n--- RAW MCP RESULT OBJECT ---")
                print(result)
                print("-----------------------------\n")

                if getattr(result, "isError", False):
                     return f"Tool returned an error: {result.content}"

                if result and result.content:
                    return "\n".join([getattr(c, 'text', str(c)) for c in result.content])
                
                return "No papers found."

    except Exception as e:
        return f"MCP Connection Error: {str(e)}"

async def main():
    response = await test_arxiv_mcp()
    print("\n--- PARSED OUTPUT ---")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())