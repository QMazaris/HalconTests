import sqlite3
from pathlib import Path
from typing import Optional
import logging

from mcp.server.fastmcp import FastMCP
from rapidfuzz import process, fuzz

DB_PATH = Path(__file__).with_name("halcon_operators.db")

# Create the FastMCP server
mcp = FastMCP("halcon-mcp-server")

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_connection():
    """Get database connection."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def validate_database() -> None:
    """Validate database connectivity and log basic stats."""
    if not DB_PATH.exists():
        logging.error("Database file not found at %s", DB_PATH)
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        count = cur.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
        logging.info("Database connected: %d operators available", count)
    except Exception as exc:
        logging.exception("Failed to query database: %s", exc)
        raise
    finally:
        con.close()


@mcp.resource("halcon://operators")
def get_operators_info() -> str:
    """Get information about the HALCON operators database."""
    con = get_connection()
    try:
        cur = con.cursor()
        count = cur.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
        return f"HALCON Operators Database contains {count} operators with syntax and documentation."
    finally:
        con.close()


@mcp.tool()
def search_halcon_operators(query: str, limit: int = 10) -> str:
    """Search HALCON operators by name or functionality.
    
    Args:
        query: Search term for HALCON operator name or functionality
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        Formatted list of matching HALCON operators with similarity scores
    """
    con = get_connection()
    try:
        cur = con.cursor()
        
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
            return f"No HALCON operators found matching '{query}'"
        
        result_text = f"Found {len(results)} HALCON operators matching '{query}':\n\n"
        for r in results:
            result_text += f"**{r['name']}** (similarity: {r['score']}%)\n"
            result_text += f"Syntax: {r['syntax'][:200]}{'...' if len(r['syntax']) > 200 else ''}\n"
            result_text += f"Description: {r['description'][:150]}{'...' if len(r['description']) > 150 else ''}\n\n"
        
        return result_text
        
    finally:
        con.close()


@mcp.tool()
def get_halcon_operator(name: str) -> str:
    """Get detailed information about a specific HALCON operator.
    
    Args:
        name: Exact name of the HALCON operator
    
    Returns:
        Complete operator information including syntax, description, and documentation URL
    """
    con = get_connection()
    try:
        cur = con.cursor()
        row = cur.execute(
            "SELECT name, url, syntax, description FROM operators WHERE name = ? COLLATE NOCASE", 
            (name,)
        ).fetchone()
        
        if not row:
            return f"HALCON operator '{name}' not found"
        
        result_text = f"**{row['name']}**\n\n"
        result_text += f"**Syntax:**\n{row['syntax'] or 'No syntax available'}\n\n"
        result_text += f"**Description:**\n{row['description'] or 'No description available'}\n\n"
        result_text += f"**Documentation URL:**\n{row['url']}\n"
        
        return result_text
        
    finally:
        con.close()


@mcp.tool()
def get_halcon_syntax(name: str) -> str:
    """Get syntax information for a HALCON operator.
    
    Args:
        name: Name of the HALCON operator
    
    Returns:
        Formatted syntax information for the operator
    """
    con = get_connection()
    try:
        cur = con.cursor()
        row = cur.execute(
            "SELECT name, syntax FROM operators WHERE name = ? COLLATE NOCASE",
            (name,)
        ).fetchone()
        
        if not row:
            return f"HALCON operator '{name}' not found"
        
        syntax = row["syntax"] or "No syntax available"
        return f"**{row['name']} Syntax:**\n\n{syntax}"
        
    finally:
        con.close()


@mcp.tool()
def list_halcon_operators(offset: int = 0, limit: int = 50) -> str:
    """List HALCON operators with optional filtering.
    
    Args:
        offset: Number of results to skip (default: 0)
        limit: Maximum number of results to return (default: 50)
    
    Returns:
        Paginated list of HALCON operators with syntax previews
    """
    con = get_connection()
    try:
        cur = con.cursor()
        
        rows = cur.execute(
            "SELECT name, syntax FROM operators ORDER BY name LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        
        total = cur.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
        
        result_text = f"HALCON Operators ({offset + 1}-{offset + len(rows)} of {total}):\n\n"
        for row in rows:
            syntax_preview = row["syntax"][:100] + "..." if row["syntax"] and len(row["syntax"]) > 100 else row["syntax"] or "No syntax"
            result_text += f"**{row['name']}**\n{syntax_preview}\n\n"
        
        return result_text
        
    finally:
        con.close()


if __name__ == "__main__":
    logging.info("Starting HALCON MCP server …")
    validate_database()
    logging.info("Launching FastMCP (transport=stdio)…")
    mcp.run(transport="stdio") 