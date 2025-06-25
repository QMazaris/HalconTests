import sqlite3
import os
import pickle
from pathlib import Path
from typing import Optional, Union
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager

from mcp.server.fastmcp import FastMCP

# Configuration Constants
DB_PATH = Path(os.getenv("HALCON_DB_PATH", Path(__file__).with_name("combined.db")))
CODE_DB_PATH = Path(os.getenv("HALCON_CODE_DB_PATH", Path(__file__).with_name("halcon_code_examplesV2.db")))

# Pre-built index paths
OPERATOR_INDEX_PATH = Path(__file__).with_name("halcon_operators.faiss")
OPERATOR_META_PATH = Path(__file__).with_name("halcon_operators_meta.pkl")
CODE_INDEX_PATH = Path(__file__).with_name("halcon_code_examples.faiss")
CODE_META_PATH = Path(__file__).with_name("halcon_code_examples_meta.pkl")

# Semantic search configuration
SEMANTIC_MODEL_NAME = os.getenv("HALCON_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_DEFAULT = 5
USE_QUANTIZATION = True
QUANTIZATION_BITS = 8
NPROBE = 16

# Create the FastMCP server
mcp = FastMCP("halcon-mcp-server")

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Global variables for semantic search
_embedding_model = None
_faiss_index = None
_operator_meta = None
_code_index = None
_code_meta = None


@contextmanager
def get_connection():
    """Context manager for database connections."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


@contextmanager
def get_code_connection():
    """Context manager for code database connections."""
    con = sqlite3.connect(CODE_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def validate_database() -> None:
    """Validate database connectivity and log basic stats for both operators and code examples."""
    # Validate Operator Database
    if not DB_PATH.exists():
        logging.error("Operator database file not found at %s", DB_PATH)
        raise FileNotFoundError(f"Operator database not found: {DB_PATH}")

    with get_connection() as con:
        try:
            cur = con.cursor()
            count = cur.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
            logging.info("Operator database connected: %d operators available", count)

            # Check schema
            columns = cur.execute("PRAGMA table_info(operators)").fetchall()
            col_names = [col[1] for col in columns]
            required_cols = ['name', 'signature', 'description', 'parameters', 'results', 'url']

            for col in required_cols:
                if col not in col_names:
                    logging.error("Missing required column in operators table: %s", col)
                    raise ValueError(f"Database schema missing column: {col}")

            logging.info("Operator database schema validated successfully")

        except Exception as exc:
            logging.exception("Failed to query operator database: %s", exc)
            raise

    # Validate Code Examples Database
    if not CODE_DB_PATH.exists():
        logging.error("Code examples database file not found at %s", CODE_DB_PATH)
        raise FileNotFoundError(f"Code examples database not found: {CODE_DB_PATH}")

    with get_code_connection() as con:
        try:
            cur = con.cursor()
            count = cur.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
            logging.info("Code examples database connected: %d examples available", count)

            # Check schema
            columns = cur.execute("PRAGMA table_info(examples)").fetchall()
            col_names = [col[1] for col in columns]
            required_cols = ['title', 'description', 'code', 'tags']

            for col in required_cols:
                if col not in col_names:
                    logging.error("Missing required column in examples table: %s", col)
                    raise ValueError(f"Database schema missing column: {col}")
            
            logging.info("Code examples database schema validated successfully")

        except Exception as exc:
            logging.exception("Failed to query code examples database: %s", exc)
            raise


def _build_faiss_index(embeddings: np.ndarray, use_quantization: bool = True) -> faiss.Index:
    """Build optimized FAISS index based on dataset size and settings."""
    dim = embeddings.shape[1]
    n_vectors = len(embeddings)
    
    if use_quantization and n_vectors > 500:
        # Use IVF (Inverted File) with inner product for cosine similarity on normalized vectors
        n_centroids = min(max(int(np.sqrt(n_vectors)), 50), n_vectors // 10)
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))
        index.nprobe = NPROBE
        logging.info("Built quantized IVF index with %d centroids", n_centroids)
    else:
        # Use flat index for small datasets or exact search
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        logging.info("Built flat index for exact search")
    
    return index


def _ensure_embedding_model() -> None:
    """Ensure the embedding model is loaded."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)


def _ensure_semantic_index() -> None:
    """Load or build the FAISS index with operator embeddings."""
    global _faiss_index, _operator_meta

    if _faiss_index is not None:
        return  # Already loaded

    # Try to load pre-built index first
    if OPERATOR_INDEX_PATH.exists() and OPERATOR_META_PATH.exists():
        logging.info("Loading pre-built semantic index for HALCON operators...")
        try:
            _faiss_index = faiss.read_index(str(OPERATOR_INDEX_PATH))
            with open(OPERATOR_META_PATH, 'rb') as f:
                _operator_meta = pickle.load(f)
            
            _ensure_embedding_model()
            
            # Set nprobe for quantized indices
            if hasattr(_faiss_index, 'nprobe'):
                _faiss_index.nprobe = NPROBE
                
            logging.info("Pre-built index loaded with %d operator embeddings", len(_operator_meta))
            return
        except Exception as e:
            logging.warning("Failed to load pre-built index, rebuilding: %s", e)

    # Build index from scratch
    logging.info("Building semantic embedding index for HALCON operators...")
    _ensure_embedding_model()

    # Fetch operator data
    with get_connection() as con:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT name, description, signature, url, parameters, results FROM operators"
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
                    "parameters": row["parameters"],
                    "results": row["results"],
                }
            )

    # Compute embeddings
    embeddings = _embedding_model.encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    )

    # Normalize to unit length for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    _faiss_index = _build_faiss_index(embeddings, USE_QUANTIZATION)
    logging.info("Semantic index built with %d operator embeddings", len(texts))

    # Save index for future runs
    try:
        logging.info("Saving semantic index for operators to %s", OPERATOR_INDEX_PATH)
        faiss.write_index(_faiss_index, str(OPERATOR_INDEX_PATH))
        with open(OPERATOR_META_PATH, 'wb') as f:
            pickle.dump(_operator_meta, f)
        logging.info("Operator index saved successfully.")
    except Exception as e:
        logging.warning("Could not save operator index: %s", e)


def _ensure_code_index() -> None:
    """Load or build the FAISS index with embeddings of code examples."""
    global _code_index, _code_meta

    if _code_index is not None:
        return  # Already loaded

    # Try to load pre-built index first
    if CODE_INDEX_PATH.exists() and CODE_META_PATH.exists():
        logging.info("Loading pre-built semantic index for HALCON code examples...")
        try:
            _code_index = faiss.read_index(str(CODE_INDEX_PATH))
            with open(CODE_META_PATH, 'rb') as f:
                _code_meta = pickle.load(f)
            
            _ensure_embedding_model()
            
            # Set nprobe for quantized indices
            if hasattr(_code_index, 'nprobe'):
                _code_index.nprobe = NPROBE
                
            logging.info("Pre-built code index loaded with %d example embeddings", len(_code_meta))
            return
        except Exception as e:
            logging.warning("Failed to load pre-built code index, rebuilding: %s", e)

    # Build index from scratch
    logging.info("Building semantic embedding index for HALCON code examples...")
    _ensure_embedding_model()

    # Fetch code example data
    with get_code_connection() as con:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT title, description, code, tags FROM examples"
        ).fetchall()

        texts: list[str] = []
        _code_meta = []

        for row in rows:
            # Combine available textual fields for embedding, including a code snippet
            code_snippet = (row["code"] or "")[:256]
            text_parts = [
                row["title"] or "",
                row["description"] or "",
                code_snippet,
            ]
            texts.append(" ".join(text_parts).strip())
            _code_meta.append(
                {
                    "title": row["title"],
                    "description": row["description"],
                    "code": row["code"],
                    "tags": row["tags"],
                }
            )

    # Compute embeddings
    embeddings = _embedding_model.encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    _code_index = _build_faiss_index(embeddings, USE_QUANTIZATION)
    logging.info("Semantic index built with %d code example embeddings", len(texts))

    # Save index for future runs
    try:
        logging.info("Saving semantic index for code examples to %s", CODE_INDEX_PATH)
        faiss.write_index(_code_index, str(CODE_INDEX_PATH))
        with open(CODE_META_PATH, 'wb') as f:
            pickle.dump(_code_meta, f)
        logging.info("Code example index saved successfully.")
    except Exception as e:
        logging.warning("Could not save code example index: %s", e)


@mcp.resource("halcon://operators")
def get_operators_info() -> str:
    """Get a summary of the HALCON knowledge base, including operator and code example counts."""
    with get_connection() as con:
        op_count = con.execute("SELECT COUNT(*) FROM operators").fetchone()[0]

    code_count = 0
    try:
        if CODE_DB_PATH.exists():
            with get_code_connection() as code_con:
                code_count = code_con.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
    except Exception:
        pass  # It's okay if code examples are not available.

    return (
        f"HALCON knowledge base ready. Contains {op_count} operators and {code_count} code examples. "
        "Use tools for semantic search, detailed operator lookup, and listing."
    )


@mcp.tool()
def get_halcon_operator(
    name: str, 
    fields: list[str] = ["name", "signature", "description", "url"]
) -> Union[dict, str]:
    """Get HALCON operator information with flexible field selection.

    Args:
        name: Exact name of the HALCON operator (case-insensitive)
        fields: List of fields to return, or ["all"] to get all fields. Available fields:
               - "name": Operator name
               - "signature": Function signature/syntax
               - "description": Operator description
               - "parameters": Input parameters details
               - "results": Output results details  
               - "url": Documentation URL
               Default: ["name", "signature", "description", "url"]

    Returns:
        Dictionary with requested fields, or error message if operator not found.
    """
    valid_fields = {"name", "signature", "description", "parameters", "results", "url"}
    
    if "all" in fields:
        requested_fields = list(valid_fields)
    else:
        requested_fields = [f for f in fields if f in valid_fields]
    
    if not requested_fields:
        return "No valid fields specified. Available fields: name, signature, description, parameters, results, url, or 'all'."
    
    # Build SQL query with only requested fields
    field_str = ", ".join(requested_fields)
    
    with get_connection() as con:
        cur = con.cursor()
        
        row = cur.execute(
            f"SELECT {field_str} FROM operators WHERE name = ? COLLATE NOCASE", 
            (name,)
        ).fetchone()
        
        if not row:
            return f"HALCON operator '{name}' not found"
        
        # Build result dictionary with only requested fields
        result = {}
        for field in requested_fields:
            result[field] = row[field] if row[field] is not None else ""
        
        return result


@mcp.tool()
def list_halcon_operators(offset: int = 0, limit: int = 50) -> str:
    """List HALCON operators with pagination.

    Args:
        offset: Number of results to skip (default: 0)
        limit: Maximum results to return (default: 50, max: 100)

    Returns:
        Paginated list of operators with brief descriptions.
    """
    with get_connection() as con:
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


@mcp.tool()
def semantic_match(
    query: str,
    k: int = TOP_K_DEFAULT,
    fields: list[str] = ["name", "signature", "description", "url"],
) -> list[dict]:
    """Return top-k semantically matching HALCON operators for a natural language query.

    Args:
        query: Natural language search string.
        k: Number of matches to return (default 5).
        fields: List of fields to return, or ["all"] to get all fields. Available fields:
               - "name": Operator name
               - "signature": Function signature/syntax
               - "description": Operator description
               - "parameters": Input parameters details
               - "results": Output results details  
               - "url": Documentation URL
               Default: ["name", "signature", "description", "url"]

    Returns:
        List of dictionaries with requested operator information and similarity score.
    """
    if not query or len(query.strip()) < 3:
        return []

    # Validate fields
    valid_fields = {"name", "signature", "description", "parameters", "results", "url"}
    
    if "all" in fields:
        requested_fields = list(valid_fields)
    else:
        requested_fields = [f for f in fields if f in valid_fields]
    
    if not requested_fields:
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
        
        # Build result with requested fields plus score
        result = {"score": float(score)}
        for field in requested_fields:
            if field in meta:
                result[field] = meta[field]
        
        results.append(result)

    return results


@mcp.tool()
def semantic_code_search(
    query: str,
    k: int = TOP_K_DEFAULT,
) -> list[dict]:
    """Return top-k code example chunks matching a natural language query.

    Each result contains title, description, tags, full code, and similarity score.
    
    Args:
        query: Natural language search string.
        k: Number of matches to return (default 5).
        
    Returns:
        List of dictionaries with code example information and similarity score.
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