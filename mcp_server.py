import asyncio
from fastmcp import Client, FastMCP

mcp = FastMCP("Image Handler")

def format_response(response: str) -> str:
    return f"Response: {response}"

@mcp.tool()
def test(text: str) -> str:
    """Test the availabilty of the Image Hander service"""
    return f"TEST SUCCESSFUL: {text}"

@mcp.tool()
def test2(text: str) -> str:
    """Test the correctness of the setup"""
    return f"TEST2 SUCCESSFUL: {text}"

if __name__ == "__main__":
    asyncio.run(main())