import sqlite3
import os
from pathlib import Path
from typing import Literal
import logging

from mcp.server.fastmcp import FastMCP
from rapidfuzz import process, fuzz

DB_PATH = Path(os.getenv("HALCON_DB_PATH", Path(__file__).with_name("halcon_operators_new.db")))

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
        
        # Check if we have the new schema
        columns = cur.execute("PRAGMA table_info(operators)").fetchall()
        col_names = [col[1] for col in columns]
        required_cols = ['name', 'signature', 'description', 'page_dump', 'url']
        
        for col in required_cols:
            if col not in col_names:
                logging.error("Missing required column: %s", col)
                raise ValueError(f"Database schema missing column: {col}")
        
        logging.info("Database schema validated successfully")
        
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
        return f"HALCON Operators Database contains {count} operators with signature, description, page dumps, and documentation URLs."
    finally:
        con.close()


@mcp.tool()
def search_halcon_operators(query: str, limit: int = 10) -> str:
    """Search HALCON operators by name or functionality.

    Args:
        query: Search term for operator name or functionality
        limit: Maximum number of results (default: 10, max: 20)

    Returns:
        Formatted list of matching operators with similarity scores.
    """
    con = get_connection()
    try:
        cur = con.cursor()
        
        # Get all operators for fuzzy search
        rows = cur.execute("SELECT name, description FROM operators").fetchall()
        
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
                        "description": row["description"] or "No description available",
                        "score": score
                    })
        
        if not results:
            return f"No HALCON operators found matching '{query}'"
        
        result_text = f"Found {len(results)} HALCON operators matching '{query}':\n\n"
        for r in results:
            result_text += f"**{r['name']}** (similarity: {r['score']}%)\n"
            result_text += f"Description: {r['description'][:150]}{'...' if len(r['description']) > 150 else ''}\n\n"
        
        return result_text
        
    finally:
        con.close()


@mcp.tool()
def get_halcon_operator(name: str, detail: Literal["signature", "info", "full"] = "info") -> str:
    """Get HALCON operator information with different levels of detail.

    Args:
        name: Exact name of the HALCON operator (case-insensitive)
        detail: Level of detail to return:
                - "signature": Just the function signature/syntax
                - "info": Name, signature, description, and URL (default)
                - "full": Complete documentation including full page dump

    Returns:
        Operator information formatted according to detail level.
    """
    con = get_connection()
    try:
        cur = con.cursor()
        
        # Select fields based on detail level
        if detail == "signature":
            fields = "name, signature"
        elif detail == "full":
            fields = "name, signature, description, page_dump, url"
        else:  # info
            fields = "name, signature, description, url"
        
        row = cur.execute(
            f"SELECT {fields} FROM operators WHERE name = ? COLLATE NOCASE", 
            (name,)
        ).fetchone()
        
        if not row:
            return f"HALCON operator '{name}' not found"
        
        # Format response based on detail level
        if detail == "signature":
            if row['signature']:
                return f"**{row['name']}** signature:\n```\n{row['signature']}\n```"
            else:
                return f"**{row['name']}**: No signature available"
        
        elif detail == "full":
            result_text = f"**{row['name']} - Complete Documentation**\n\n"
            if row['signature']:
                result_text += f"**Signature:**\n```\n{row['signature']}\n```\n\n"
            result_text += f"**Description:**\n{row['description'] or 'No description available'}\n\n"
            result_text += f"**Source:** {row['url']}\n\n"
            result_text += "**Full Documentation Content:**\n\n"
            result_text += row['page_dump'] or 'No page dump available'
            return result_text
        
        else:  # info
            result_text = f"**{row['name']}**\n\n"
            if row['signature']:
                result_text += f"**Signature:**\n```\n{row['signature']}\n```\n\n"
            result_text += f"**Description:**\n{row['description'] or 'No description available'}\n\n"
            result_text += f"**Documentation URL:**\n{row['url']}\n"
            return result_text
        
    finally:
        con.close()


@mcp.tool()
def list_halcon_operators(offset: int = 0, limit: int = 50) -> str:
    """List HALCON operators with pagination.

    Args:
        offset: Number of results to skip (default: 0)
        limit: Maximum results to return (default: 50, max: 100)

    Returns:
        Paginated list of operators with brief descriptions.
    """
    con = get_connection()
    try:
        cur = con.cursor()
        
        rows = cur.execute(
            "SELECT name, description FROM operators ORDER BY name LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        
        total = cur.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
        
        result_text = f"HALCON Operators ({offset + 1}-{offset + len(rows)} of {total}):\n\n"
        for row in rows:
            description_preview = row["description"][:100] + "..." if row["description"] and len(row["description"]) > 100 else row["description"] or "No description available"
            result_text += f"**{row['name']}**\n{description_preview}\n\n"
        
        return result_text
        
    finally:
        con.close()


if __name__ == "__main__":
    logging.info("Starting HALCON MCP server …")
    validate_database()
    logging.info("Launching FastMCP (transport=stdio)…")
    mcp.run(transport="stdio") 