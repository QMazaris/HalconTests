import sqlite3
import os
import pickle
from pathlib import Path
from typing import Literal
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from mcp.server.fastmcp import FastMCP
from rapidfuzz import process, fuzz

DB_PATH = Path(os.getenv("HALCON_DB_PATH", Path(__file__).with_name("combined.db")))
# New: path to the code examples database (populated via chunk_scanner_cli.py)
CODE_DB_PATH = Path(os.getenv("HALCON_CODE_DB_PATH", Path(__file__).with_name("halcon_code_examplesV2.db")))

# Pre-built index paths
OPERATOR_INDEX_PATH = Path(__file__).with_name("halcon_operators.faiss")
OPERATOR_META_PATH = Path(__file__).with_name("halcon_operators_meta.pkl")
CODE_INDEX_PATH = Path(__file__).with_name("halcon_code_examples.faiss")
CODE_META_PATH = Path(__file__).with_name("halcon_code_examples_meta.pkl")

# Create the FastMCP server
mcp = FastMCP("halcon-mcp-server")

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_connection():
    """Get database connection."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def get_code_con():
    """Return SQLite connection to the HALCON code examples database."""
    con = sqlite3.connect(CODE_DB_PATH, check_same_thread=False)
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
        required_cols = ['name', 'signature', 'description', 'parameters', 'results', 'url']
        
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
            fields = "name, signature, description, url"
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
            result_text += f"**Source:** {row['url']}\n"
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


# ------------------------------------------------------------
# Semantic operator matching (/semantic_match)
# ------------------------------------------------------------

# Configuration constants
SEMANTIC_MODEL_NAME = os.getenv("HALCON_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_TOP_K_DEFAULT = 5

# Quantization settings for performance
USE_QUANTIZATION = True  # Set to False for exact search, True for faster approximate search
QUANTIZATION_BITS = 8    # 8-bit quantization for good speed/accuracy tradeoff
NPROBE = 16             # Number of clusters to search (higher = more accurate but slower)

# Global variables for semantic search
_embedding_model = None
_faiss_index = None
_operator_meta = None
_code_index = None
_code_meta = None


def _ensure_semantic_index() -> None:
    """Load or build the FAISS index with operator embeddings."""
    global _embedding_model, _faiss_index, _operator_meta

    if _faiss_index is not None:
        return  # Already loaded

    # Try to load pre-built index first
    if OPERATOR_INDEX_PATH.exists() and OPERATOR_META_PATH.exists():
        logging.info("Loading pre-built semantic index for HALCON operators...")
        try:
            _faiss_index = faiss.read_index(str(OPERATOR_INDEX_PATH))
            with open(OPERATOR_META_PATH, 'rb') as f:
                _operator_meta = pickle.load(f)
            
            # Load embedding model for query encoding
            _embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
            
            # Set nprobe for quantized indices
            if hasattr(_faiss_index, 'nprobe'):
                _faiss_index.nprobe = NPROBE
                
            logging.info("Pre-built index loaded with %d operator embeddings", len(_operator_meta))
            return
        except Exception as e:
            logging.warning("Failed to load pre-built index, rebuilding: %s", e)

    # Build index from scratch
    logging.info("Building semantic embedding index for HALCON operators...")
    _embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

    # Fetch operator data
    con = get_connection()
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT name, description, signature, url FROM operators"
        ).fetchall()

        texts: list[str] = []
        _operator_meta = []

        for row in rows:
            text = row["description"] or "No description available"
            texts.append(text)
            _operator_meta.append(
                {
                    "name": row["name"],
                    "description": row["description"] or "No description available",
                    "signature": row["signature"],
                    "url": row["url"],
                }
            )
    finally:
        con.close()

    # Compute embeddings
    embeddings = _embedding_model.encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    )

    # Normalize to unit length for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dim = embeddings.shape[1]
    n_vectors = len(embeddings)

    # Create optimized index based on dataset size and settings
    if USE_QUANTIZATION and n_vectors > 1000:
        # Use IVF (Inverted File) with inner product for cosine similarity on normalized vectors
        n_centroids = min(max(int(np.sqrt(n_vectors)), 100), n_vectors // 10)
        _faiss_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
        _faiss_index.train(embeddings.astype(np.float32))
        _faiss_index.add(embeddings.astype(np.float32))
        _faiss_index.nprobe = NPROBE
        logging.info("Built quantized IVF index with %d centroids", n_centroids)
    else:
        # Use flat index for small datasets or exact search
        _faiss_index = faiss.IndexFlatIP(dim)
        _faiss_index.add(embeddings.astype(np.float32))
        logging.info("Built flat index for exact search")

    logging.info("Semantic index built with %d operator embeddings", len(texts))


# ------------------------------------------------------------
# Semantic code example matching (internal helper)
# ------------------------------------------------------------

def _ensure_code_index() -> None:
    """Load or build the FAISS index with embeddings of code examples."""
    global _embedding_model, _code_index, _code_meta

    if _code_index is not None:
        return  # Already loaded

    # Try to load pre-built index first
    if CODE_INDEX_PATH.exists() and CODE_META_PATH.exists():
        logging.info("Loading pre-built semantic index for HALCON code examples...")
        try:
            _code_index = faiss.read_index(str(CODE_INDEX_PATH))
            with open(CODE_META_PATH, 'rb') as f:
                _code_meta = pickle.load(f)
            
            # Load embedding model for query encoding (reuse if already loaded)
            if _embedding_model is None:
                _embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
            
            # Set nprobe for quantized indices
            if hasattr(_code_index, 'nprobe'):
                _code_index.nprobe = NPROBE
                
            logging.info("Pre-built code index loaded with %d example embeddings", len(_code_meta))
            return
        except Exception as e:
            logging.warning("Failed to load pre-built code index, rebuilding: %s", e)

    # Build index from scratch
    logging.info("Building semantic embedding index for HALCON code examples...")

    # Load embedding model (reuse if already loaded)
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

    # Fetch code example data
    con = get_code_con()
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT title, description, code, tags FROM examples"
        ).fetchall()

        texts: list[str] = []
        _code_meta = []

        for row in rows:
            # Combine available textual fields for embedding, excluding code content
            text_parts = [row["title"] or "", row["description"] or ""]
            texts.append(" ".join(text_parts))
            _code_meta.append(
                {
                    "title": row["title"],
                    "description": row["description"],
                    "code": row["code"],
                    "tags": row["tags"],
                }
            )
    finally:
        con.close()

    # Compute embeddings
    embeddings = _embedding_model.encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dim = embeddings.shape[1]
    n_vectors = len(embeddings)

    # Create optimized index based on dataset size and settings
    if USE_QUANTIZATION and n_vectors > 500:
        # Use IVF for code examples (smaller dataset, fewer centroids)
        n_centroids = min(max(int(np.sqrt(n_vectors)), 50), n_vectors // 5)
        _code_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
        _code_index.train(embeddings.astype(np.float32))
        _code_index.add(embeddings.astype(np.float32))
        _code_index.nprobe = NPROBE
        logging.info("Built quantized code index with %d centroids", n_centroids)
    else:
        # Use flat index for small datasets
        _code_index = faiss.IndexFlatIP(dim)
        _code_index.add(embeddings.astype(np.float32))
        logging.info("Built flat code index for exact search")

    logging.info("Semantic index built with %d code example embeddings", len(texts))


@mcp.tool()
def semantic_match(
    query: str,
    k: int = _TOP_K_DEFAULT,
    detail: Literal["signature", "info", "full"] = "info",
):
    """Return top-k semantically matching HALCON operators for a natural language query.

    Args:
        query: Natural language search string.
        k: Number of matches to return (default 5).
        detail: Level of detail to return:
                - "signature": Just the function signature/syntax
                - "info": Name, signature, description, and URL (default)
                - "full": Complete documentation including full page dump

    Returns:
        List of dictionaries with operator information and similarity score.
    """

    if not query or len(query.strip()) < 3:
        return []

    _ensure_semantic_index()

    # Embed and normalise query
    vec = _embedding_model.encode([query], convert_to_numpy=True)
    vec = vec / np.linalg.norm(vec)

    D, I = _faiss_index.search(vec.astype(np.float32), k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        meta = _operator_meta[idx]
        op: dict = {
            "name": meta["name"],
            "score": float(score),
        }

        if detail == "signature":
            op["signature"] = meta["signature"]
        elif detail == "full":
            op.update(
                {
                    "signature": meta["signature"],
                    "description": meta["description"],
                    "url": meta["url"],
                }
            )
        else:  # info (default)
            op.update(
                {
                    "signature": meta["signature"],
                    "description": meta["description"],
                    "url": meta["url"],
                }
            )

        results.append(op)

    return results


@mcp.tool()
def semantic_code_search(
    query: str,
    k: int = _TOP_K_DEFAULT,
) -> list[dict]:
    """Return top-k code example chunks matching a natural language query.

    Each result contains title, description, tags, full code, and similarity score.
    """

    if not query or len(query.strip()) < 3:
        return []

    _ensure_code_index()

    vec = _embedding_model.encode([query], convert_to_numpy=True)
    vec = vec / np.linalg.norm(vec)

    D, I = _code_index.search(vec.astype(np.float32), k)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        meta = _code_meta[idx]
        results.append({
            "title": meta["title"],
            "description": meta["description"],
            "tags": meta["tags"],
            "code": meta["code"],
            "score": float(score),
        })

    return results


if __name__ == "__main__":
    logging.info("Starting HALCON MCP server …")
    validate_database()
    logging.info("Launching FastMCP (transport=stdio)…")
    mcp.run(transport="stdio") 