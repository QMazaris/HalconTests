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

    Use this tool when you need to find HALCON operators that match a search term.
    This performs fuzzy matching on both operator names and descriptions, so you can search
    for partial names, functionality keywords, or general concepts.

    Args:
        query: Search term for HALCON operator name or functionality. Can be:
               - Partial operator name (e.g., "read_image" to find "read_image")
               - Functionality keyword (e.g., "morphology", "filter", "blob")
               - General concept (e.g., "image processing", "measurement")
        limit: Maximum number of results to return (default: 10, max recommended: 20)

    Returns:
        Formatted list of matching HALCON operators with similarity scores.
        Each result includes the operator name, description preview, and match percentage.
        Returns "No HALCON operators found..." if no matches above 30% similarity.

    Example usage:
        - To find image reading functions: query="read image"
        - To find morphological operations: query="morphology"
        - To find a specific operator: query="threshold"
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
def get_halcon_operator(name: str) -> str:
    """Get detailed information about a specific HALCON operator.

    Use this tool when you have the exact name of a HALCON operator and need complete
    information including description and documentation URL. If you're unsure of the
    exact name, use search_halcon_operators first.

    Args:
        name: Exact name of the HALCON operator (case-insensitive).
              Must be the precise operator name as it appears in HALCON documentation.
              Examples: "read_image", "threshold", "connection", "select_shape"

    Returns:
        Complete operator information including:
        - Operator name (formatted)
        - Full description
        - Official documentation URL

    Returns "HALCON operator 'name' not found" if the operator doesn't exist.

    Example usage:
        - name="read_image" (gets info about the image reading operator)
        - name="threshold" (gets info about the thresholding operator)
    """
    con = get_connection()
    try:
        cur = con.cursor()
        row = cur.execute(
            "SELECT name, url, description FROM operators WHERE name = ? COLLATE NOCASE", 
            (name,)
        ).fetchone()
        
        if not row:
            return f"HALCON operator '{name}' not found"
        
        result_text = f"**{row['name']}**\n\n"
        result_text += f"**Description:**\n{row['description'] or 'No description available'}\n\n"
        result_text += f"**Documentation URL:**\n{row['url']}\n"
        
        return result_text
        
    finally:
        con.close()

@mcp.tool()
def list_halcon_operators(offset: int = 0, limit: int = 50) -> str:
    """List HALCON operators with pagination.

    Use this tool to browse through the available HALCON operators in alphabetical order.
    This is useful for discovering operators or getting an overview of what's available.
    For searching specific functionality, use search_halcon_operators instead.

    Args:
        offset: Number of results to skip (default: 0). Use for pagination.
                Examples: 0 (first page), 50 (second page), 100 (third page)
        limit: Maximum number of results to return (default: 50, max recommended: 100).
               Larger values may result in very long responses.

    Returns:
        Paginated list of HALCON operators showing:
        - Current page info (showing X-Y of total)
        - Operator names in alphabetical order
        - Brief syntax preview for each operator (if available)

    Example usage:
        - offset=0, limit=20 (first 20 operators)
        - offset=50, limit=25 (operators 51-75)
        - offset=0, limit=100 (first 100 operators for broad overview)
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