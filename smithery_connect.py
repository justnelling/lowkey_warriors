'''
this script connects with our node express server to interact with smithery's MCP servers
'''
import aiohttp
import asyncio
from typing import Dict, Any

class MCPBridge:
    def __init__(self, base_url: str = 'http://localhost:3000'):
        self.base_url = base_url 

    async def initialize(self, smithery_url: str) -> Dict:
        '''Initialize MCP client with specific smithery MCP server URL'''
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{self.base_url}/init',
                json={"smithery_url": smithery_url}
            ) as response:
                return await response.json()

    async def list_tools(self) -> list: 
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/tools") as response:
                return await response.json()
            
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any: 
        async with aiohttp.ClientSession() as session: 
            async with session.post(
                f"{self.base_url}/tools/{tool_name}",
                json={"params":params}
            ) as response: 
                return await response.json() 
            
async def main():
    smithery_url = ""
    mcp = MCPBridge()

    await mcp.initialize(smithery_url)

    # list tools
    tools = await mcp.list_tools() 

    print(f"Available MCP tools: {tools}")

    # call tools 

    tool_name = ""
    # dictionary mapping of params -> values
    params_dict = {}
    result = await mcp.call_tool(tool_name, params_dict)

    print(f"Tool result: {result}")
    
if __name__ == "__main__":
    asyncio.run(main())