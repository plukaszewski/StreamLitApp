from mcp.server.fastmcp import FastMCP

###########MCP###########

mcp = FastMCP("Image Handler")

def format_response(response: str) -> str:
    return f"Response: {response}"

@mcp.tool()
def test(text: str) -> str:
    """Test the availabilty of the Image Hander service"""
    return f"TEST SUCCESSFUL: {text}"

mcp.run(transport='stdio')

#########################