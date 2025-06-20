import asyncio
import sqlite3
from pathlib import Path
from typing import List, Optional, Any

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource
)
from rapidfuzz import process, fuzz

DB_PATH = Path(__file__).with_name("halcon_operators.db")

server = Server("halcon-mcp-server")


def get_connection():
    """Get database connection."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available HALCON operator resources."""
    return [
        Resource(
            uri="halcon://operators",
            name="HALCON Operators Database",
            description="Complete database of HALCON operators with syntax and documentation",
            mimeType="application/json"
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read HALCON operator resource."""
    if uri == "halcon://operators":
        con = get_connection()
        cur = con.cursor()
        count = cur.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
        con.close()
        return f"HALCON Operators Database contains {count} operators with syntax and documentation."
    raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available HALCON MCP tools."""
    return [
        Tool(
            name="search_halcon_operators",
            description="Search HALCON operators by name or functionality",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term for HALCON operator name or functionality"
                    },
                    "limit": {
                        "type": "integer", 
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_halcon_operator",
            description="Get detailed information about a specific HALCON operator",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact name of the HALCON operator"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="get_halcon_syntax",
            description="Get syntax information for a HALCON operator",
            inputSchema={
                "type": "object", 
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the HALCON operator"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="list_halcon_operators",
            description="List HALCON operators with optional filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "offset": {
                        "type": "integer",
                        "description": "Number of results to skip",
                        "default": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return", 
                        "default": 50
                    }
                },
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls for HALCON operators."""
    con = get_connection()
    cur = con.cursor()
    
    try:
        if name == "search_halcon_operators":
            query = arguments["query"]
            limit = arguments.get("limit", 10)
            
            # Get all operators for fuzzy search
            rows = cur.execute("SELECT name, syntax, description FROM operators").fetchall()
            
            # Fuzzy search on names and descriptions
            choices = {}
            for row in rows:
                choices[row["name"]] = row
                # Also search descriptions
                if row["description"]:
                    choices[f"{row['name']} - {row['description'][:100]}"] = row
            
            best = process.extract(query, choices.keys(), scorer=fuzz.WRatio, limit=limit)
            
            results = []
            seen = set()
            for match, score, _ in best:
                if score > 30:  # Minimum similarity threshold
                    row = choices[match]
                    if row["name"] not in seen:
                        seen.add(row["name"])
                        results.append({
                            "name": row["name"],
                            "syntax": row["syntax"] or "No syntax available",
                            "description": row["description"] or "No description available",
                            "score": score
                        })
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"No HALCON operators found matching '{query}'"
                )]
            
            result_text = f"Found {len(results)} HALCON operators matching '{query}':\n\n"
            for r in results:
                result_text += f"**{r['name']}** (similarity: {r['score']}%)\n"
                result_text += f"Syntax: {r['syntax'][:200]}{'...' if len(r['syntax']) > 200 else ''}\n"
                result_text += f"Description: {r['description'][:150]}{'...' if len(r['description']) > 150 else ''}\n\n"
            
            return [TextContent(type="text", text=result_text)]
            
        elif name == "get_halcon_operator":
            operator_name = arguments["name"]
            row = cur.execute(
                "SELECT name, url, syntax, description FROM operators WHERE name = ? COLLATE NOCASE", 
                (operator_name,)
            ).fetchone()
            
            if not row:
                return [TextContent(
                    type="text",
                    text=f"HALCON operator '{operator_name}' not found"
                )]
            
            result_text = f"**{row['name']}**\n\n"
            result_text += f"**Syntax:**\n{row['syntax'] or 'No syntax available'}\n\n"
            result_text += f"**Description:**\n{row['description'] or 'No description available'}\n\n"
            result_text += f"**Documentation URL:**\n{row['url']}\n"
            
            return [TextContent(type="text", text=result_text)]
            
        elif name == "get_halcon_syntax":
            operator_name = arguments["name"]
            row = cur.execute(
                "SELECT name, syntax FROM operators WHERE name = ? COLLATE NOCASE",
                (operator_name,)
            ).fetchone()
            
            if not row:
                return [TextContent(
                    type="text", 
                    text=f"HALCON operator '{operator_name}' not found"
                )]
            
            syntax = row["syntax"] or "No syntax available"
            return [TextContent(
                type="text",
                text=f"**{row['name']} Syntax:**\n\n{syntax}"
            )]
            
        elif name == "list_halcon_operators":
            offset = arguments.get("offset", 0)
            limit = arguments.get("limit", 50)
            
            rows = cur.execute(
                "SELECT name, syntax FROM operators ORDER BY name LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            
            total = cur.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
            
            result_text = f"HALCON Operators ({offset + 1}-{offset + len(rows)} of {total}):\n\n"
            for row in rows:
                syntax_preview = row["syntax"][:100] + "..." if row["syntax"] and len(row["syntax"]) > 100 else row["syntax"] or "No syntax"
                result_text += f"**{row['name']}**\n{syntax_preview}\n\n"
            
            return [TextContent(type="text", text=result_text)]
            
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
            
    finally:
        con.close()


async def main():
    """Main entry point for the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="halcon-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main()) 