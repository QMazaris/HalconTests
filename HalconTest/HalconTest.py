import sqlite3
import os
import pickle
from pathlib import Path
import sys
from typing import Optional, Union
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
import subprocess
import sys

from mcp.server.fastmcp import FastMCP

# Configuration Constants
SCRIPT_DIR = Path(__file__).parent
DB_DIR = SCRIPT_DIR / "databases"
DB_PATH = Path(os.getenv("HALCON_DB_PATH", DB_DIR / "combined.db"))
CHUNK_DB_PATH = Path(os.getenv("HALCON_CHUNK_DB_PATH", DB_DIR / "halcon_chunks_latest.db"))

# Pre-built index paths (also in databases folder)
OPERATOR_INDEX_PATH = DB_DIR / "halcon_operators.faiss"
OPERATOR_META_PATH = DB_DIR / "halcon_operators_meta.pkl"
CHUNK_INDEX_PATH = DB_DIR / "halcon_chunks.faiss"
CHUNK_META_PATH = DB_DIR / "halcon_chunks_meta.pkl"

# New index paths for separated chunk types
FULL_CHUNK_INDEX_PATH = DB_DIR / "halcon_chunks_full.faiss"
FULL_CHUNK_META_PATH = DB_DIR / "halcon_chunks_full_meta.pkl"
MICRO_CHUNK_INDEX_PATH = DB_DIR / "halcon_chunks_micro.faiss"
MICRO_CHUNK_META_PATH = DB_DIR / "halcon_chunks_micro_meta.pkl"

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
_full_chunk_index = None
_full_chunk_meta = None
_micro_chunk_index = None
_micro_chunk_meta = None
_build_script_was_run = False


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
def get_chunk_connection():
    """Context manager for chunk database connections."""
    con = sqlite3.connect(CHUNK_DB_PATH, check_same_thread=False)
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

    # Validate Chunk Database (optional - may not exist if chunks haven't been generated)
    if CHUNK_DB_PATH.exists():
        with get_chunk_connection() as con:
            try:
                cur = con.cursor()
                
                # Check if both tables exist
                tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                table_names = [table[0] for table in tables]
                
                if 'contexts' in table_names and 'chunks' in table_names:
                    context_count = cur.execute("SELECT COUNT(*) FROM contexts").fetchone()[0]
                    chunk_count = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                    logging.info("Chunk database connected: %d contexts, %d chunks available", context_count, chunk_count)

                    # Check contexts schema
                    columns = cur.execute("PRAGMA table_info(contexts)").fetchall()
                    col_names = [col[1] for col in columns]
                    required_cols = ['id', 'file', 'procedure', 'header', 'tags']

                    for col in required_cols:
                        if col not in col_names:
                            logging.warning("Missing column in contexts table: %s", col)

                    # Check chunks schema
                    columns = cur.execute("PRAGMA table_info(chunks)").fetchall()
                    col_names = [col[1] for col in columns]
                    required_cols = ['context_id', 'chunk_type', 'sequence', 'description', 'code', 'line_start', 'line_end']

                    for col in required_cols:
                        if col not in col_names:
                            logging.warning("Missing column in chunks table: %s", col)
                    
                    logging.info("Chunk database schema validated successfully")
                else:
                    logging.warning("Chunk database exists but missing required tables")

            except Exception as exc:
                logging.warning("Failed to validate chunk database: %s", exc)
    else:
        logging.info("Chunk database not found - chunk search features will be unavailable")


def _run_build_script() -> bool:
    """Run the build_semantic_indices.py script if it hasn't been run yet."""
    global _build_script_was_run
    if _build_script_was_run:
        logging.error("Build script was already run, but required index files are still missing.")
        return False

    build_script_path = SCRIPT_DIR / "build_semantic_indices.py"
    if not build_script_path.exists():
        logging.error("Build script not found at %s", build_script_path)
        raise FileNotFoundError(f"Build script not found: {build_script_path}")
    
    logging.info("Attempting to build indices by running %s...", build_script_path.name)
    try:
        # Use sys.executable to ensure we run with the same Python interpreter
        result = subprocess.run(
            [sys.executable, str(build_script_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        logging.info("Build script completed successfully.")
        logging.debug("Build script output:\n%s", result.stdout)
        _build_script_was_run = True
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Failed to build semantic indices. Script exited with error.")
        logging.error("STDOUT:\n%s", e.stdout)
        logging.error("STDERR:\n%s", e.stderr)
        _build_script_was_run = True  # Mark as run even on failure to prevent loops
        return False
    except FileNotFoundError:
        logging.error("Failed to run build script. Is Python installed and in the system's PATH?")
        return False


def _ensure_embedding_model() -> None:
    """Ensure the embedding model is loaded."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)


def _embed_text(text: str, is_query: bool = False) -> np.ndarray:
    """Embed text optimized for the selected model."""
    _ensure_embedding_model()
    

    embedding = _embedding_model.encode([text], convert_to_numpy=True)
    return embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity


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
            logging.warning("Failed to load pre-built index, will attempt to rebuild: %s", e)

    # If loading fails or files don't exist, run the build script
    logging.warning("Operator index not found or failed to load. Attempting to build all indices...")
    if _run_build_script():
        if _faiss_index is None: # Check if another thread already loaded it
             _ensure_semantic_index() # Recursive call to retry loading
    else:
        raise RuntimeError("Failed to build and load operator semantic index.")


def _ensure_full_chunk_index() -> None:
    """Load or build the FAISS index for FULL chunks (chunk_type='full')."""
    global _full_chunk_index, _full_chunk_meta

    if _full_chunk_index is not None:
        return  # Already loaded

    # Try to load pre-built index first
    if FULL_CHUNK_INDEX_PATH.exists() and FULL_CHUNK_META_PATH.exists():
        logging.info("Loading pre-built semantic index for HALCON FULL code chunks…")
        try:
            _full_chunk_index = faiss.read_index(str(FULL_CHUNK_INDEX_PATH))
            with open(FULL_CHUNK_META_PATH, 'rb') as f:
                _full_chunk_meta = pickle.load(f)
            _ensure_embedding_model()
            if hasattr(_full_chunk_index, 'nprobe'):
                _full_chunk_index.nprobe = NPROBE
            logging.info("Pre-built FULL chunk index loaded with %d embeddings", len(_full_chunk_meta))
            return
        except Exception as e:
            logging.warning("Failed to load FULL chunk index, will attempt to rebuild: %s", e)

    # If loading fails or files don't exist, run the build script
    logging.warning("FULL chunk index not found or failed to load. Attempting to build all indices...")
    if _run_build_script():
        if _full_chunk_index is None:
            _ensure_full_chunk_index() # Recursive call
    else:
        raise RuntimeError("Failed to build and load FULL chunk semantic index.")


def _ensure_micro_chunk_index() -> None:
    """Load or build the FAISS index for MICRO chunks (chunk_type='micro')."""
    global _micro_chunk_index, _micro_chunk_meta

    if _micro_chunk_index is not None:
        return  # Already loaded

    # Try to load pre-built index first
    if MICRO_CHUNK_INDEX_PATH.exists() and MICRO_CHUNK_META_PATH.exists():
        logging.info("Loading pre-built semantic index for HALCON MICRO code chunks…")
        try:
            _micro_chunk_index = faiss.read_index(str(MICRO_CHUNK_INDEX_PATH))
            with open(MICRO_CHUNK_META_PATH, 'rb') as f:
                _micro_chunk_meta = pickle.load(f)
            _ensure_embedding_model()
            if hasattr(_micro_chunk_index, 'nprobe'):
                _micro_chunk_index.nprobe = NPROBE
            logging.info("Pre-built MICRO chunk index loaded with %d embeddings", len(_micro_chunk_meta))
            return
        except Exception as e:
            logging.warning("Failed to load MICRO chunk index, will attempt to rebuild: %s", e)

    # If loading fails or files don't exist, run the build script
    logging.warning("MICRO chunk index not found or failed to load. Attempting to build all indices...")
    if _run_build_script():
        if _micro_chunk_index is None:
            _ensure_micro_chunk_index() # Recursive call
    else:
        raise RuntimeError("Failed to build and load MICRO chunk semantic index.")


@mcp.resource("halcon://operators")
def get_operators_info() -> str:
    """Get a summary of the HALCON knowledge base, including operator, code example, and chunk counts."""
    with get_connection() as con:
        op_count = con.execute("SELECT COUNT(*) FROM operators").fetchone()[0]

    chunk_info = ""
    try:
        if CHUNK_DB_PATH.exists():
            with get_chunk_connection() as chunk_con:
                context_count = chunk_con.execute("SELECT COUNT(*) FROM contexts").fetchone()[0]
                chunk_count = chunk_con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                chunk_info = f", {context_count} file contexts, and {chunk_count} code chunks"
    except Exception:
        pass  # It's okay if chunks are not available.

    return (
        f"HALCON knowledge base ready. Contains {op_count} operators{chunk_info}. "
        "Use tools for semantic search, detailed operator lookup, and enhanced chunk search with context injection."
    )


@mcp.tool()
def search_operators(
    query: str,
    fields: list[str] = ["name", "signature", "description", "url"],
    k: int = TOP_K_DEFAULT
) -> Union[dict, list[dict], str]:
    """Unified HALCON operator search: single-word exact-first, multi-word semantic-only."""
    try:
        if not query or len(query.strip()) < 2:
            return "Query too short. Please provide at least 2 characters."

        query = query.strip()
        
        # Validate fields
        valid_fields = {"name", "signature", "description", "parameters", "results", "url"}
        
        if "all" in fields:
            requested_fields = list(valid_fields)
        else:
            requested_fields = [f for f in fields if f in valid_fields]
        
        if not requested_fields:
            return "No valid fields specified. Available fields: name, signature, description, parameters, results, url, or 'all'."

        # Decide strategy: exact-first for single-word queries, else semantic-only
        single_word = len(query.split()) == 1
        field_str = ", ".join(requested_fields)
        if single_word:
            with get_connection() as con:
                cur = con.cursor()
                row = cur.execute(
                    f"SELECT {field_str} FROM operators WHERE name = ? COLLATE NOCASE",
                    (query,)
                ).fetchone()
            if row:
                result = {field: (row[field] or "") for field in requested_fields}
                result["search_mode_used"] = "exact"
                return result

        # Semantic search (for multi-word queries or single-word fallback)
        _ensure_semantic_index()

        # Embed query for semantic search
        vec = _embed_text(query, is_query=True)

        D, I = _faiss_index.search(vec.astype(np.float32), k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0:
                continue
            meta = _operator_meta[idx]
            
            # Build result with requested fields plus score
            result = {"score": float(score), "search_mode_used": "semantic"}
            for field in requested_fields:
                if field in meta:
                    result[field] = meta[field]
            
            results.append(result)

        if results:
            return results
        else:
            return f"No operators found matching '{query}'. Try different keywords or check spelling."
    
    except Exception as e:
        logging.exception("Error in search_operators: %s", e)
        return f"Search failed: {str(e)}"


@mcp.tool()
def search_code(
    query: Optional[str] = None,
    chunk_type: str = "all",    # "full", "micro", "all"
    include_context: bool = True,
    include_navigation: bool = True,
    k: int = TOP_K_DEFAULT,
    chunk_id: Optional[int] = None,  # Direct chunk ID lookup for navigation
    direction: Optional[str] = None  # "previous" or "next" for navigation
) -> list[dict]:
    """Perform semantic code search or navigate by chunk ID.

    Uses pre-built FAISS indices for semantic search, or direct database lookup
    for chunk ID navigation. Navigation stays within the same file/context.

    Args:
        query: Natural language search string (min length 3). Ignored if chunk_id provided.
        chunk_type: Which index to query: "full", "micro", or "all".
        include_context: For micro chunks, inject surrounding context in results.
        include_navigation: Include previous/next pointers in returned results.
        k: Number of top matches to return for semantic search.
        chunk_id: Direct chunk ID to retrieve (for navigation).
        direction: Optional "previous" or "next" to get adjacent chunk.
    """
    try:
        # Handle direct chunk ID lookup (navigation mode)
        if chunk_id is not None:
            return _get_chunk_by_id(chunk_id, direction, chunk_type, include_context, include_navigation)

        # Handle semantic search
        if not query or len(query.strip()) < 3:
            return []

        query = query.strip()

        # Ensure relevant indices are loaded
        if chunk_type in ("full", "all"):
            _ensure_full_chunk_index()
        if chunk_type in ("micro", "all"):
            _ensure_micro_chunk_index()

        all_results: list[dict] = []

        # Helper to search a specific index
        def _search_index(index, meta, source_name: str):
            vec = _embed_text(query, is_query=True)
            D, I = index.search(vec.astype(np.float32), k)
            for idx, score in zip(I[0], D[0]):
                if idx < 0:
                    continue
                m = meta[idx]
                result = _build_chunk_result(m, source_name, float(score), include_context, include_navigation)
                all_results.append(result)

        if chunk_type in ("full", "all") and _full_chunk_index is not None:
            _search_index(_full_chunk_index, _full_chunk_meta, "full")
        if chunk_type in ("micro", "all") and _micro_chunk_index is not None:
            _search_index(_micro_chunk_index, _micro_chunk_meta, "micro")

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:k]

    except Exception as e:
        logging.exception("Error in search_code: %s", e)
        return [{"error": f"Search failed: {str(e)}"}]


def _get_chunk_by_id(chunk_id: int, direction: Optional[str], chunk_type: str, 
                     include_context: bool, include_navigation: bool) -> list[dict]:
    """Get a specific chunk by ID, optionally navigating to adjacent chunk."""
    with get_chunk_connection() as con:
        cur = con.cursor()
        
        target_chunk_id = chunk_id
        
        if direction in ("previous", "next"):
            # Get current chunk to find its context and sequence
            current_row = cur.execute(
                """SELECT c.context_id, c.sequence FROM chunks c WHERE c.id = ?""", 
                (chunk_id,)
            ).fetchone()
            
            if not current_row:
                return []
            
            context_id = current_row["context_id"]
            current_sequence = current_row["sequence"]
            
            # Find adjacent chunk in same context, regardless of type
            if direction == "next":
                adjacent_row = cur.execute(
                    """SELECT c.id FROM chunks c 
                       WHERE c.context_id = ? AND c.sequence > ?
                       ORDER BY c.sequence ASC LIMIT 1""",
                    (context_id, current_sequence)
                ).fetchone()
            else:  # previous
                adjacent_row = cur.execute(
                    """SELECT c.id FROM chunks c 
                       WHERE c.context_id = ? AND c.sequence < ?
                       ORDER BY c.sequence DESC LIMIT 1""",
                    (context_id, current_sequence)
                ).fetchone()
            
            if not adjacent_row:
                return []  # No adjacent chunk in same file
            
            target_chunk_id = adjacent_row["id"]
        
        # Get the full chunk data
        chunk_row = cur.execute(
            """SELECT c.id as chunk_id, c.context_id, c.chunk_type, c.sequence, 
                      c.description, c.code, c.line_start, c.line_end, c.injected_context,
                      ctx.file, ctx.procedure, ctx.header, ctx.tags
               FROM chunks c
               JOIN contexts ctx ON c.context_id = ctx.id
               WHERE c.id = ?""",
            (target_chunk_id,)
        ).fetchone()
        
        if not chunk_row:
            return []
        
        # Build result
        result = _build_chunk_result(chunk_row, chunk_row["chunk_type"], None, include_context, include_navigation)
        return [result]


def _build_chunk_result(chunk_data, source_name: str, score: Optional[float], 
                       include_context: bool, include_navigation: bool) -> dict:
    """Build a standardized chunk result dictionary."""
    result = {
        "source": source_name,
        "score": score,
        "chunk_id": chunk_data["chunk_id"],
        "context_id": chunk_data["context_id"],
        "chunk_type": chunk_data["chunk_type"],
        "sequence": chunk_data["sequence"],
        "file": chunk_data["file"],
        "procedure": chunk_data["procedure"],
        "description": chunk_data["description"],
        "code": chunk_data["code"],
        "line_start": chunk_data["line_start"],
        "line_end": chunk_data["line_end"],
        "file_header": chunk_data["header"],
        "tags": chunk_data["tags"]
    }
    
    # Add injected context for micro chunks
    if (source_name == "micro" and include_context and 
        "injected_context" in chunk_data.keys() and chunk_data["injected_context"]):
        result["injected_context"] = chunk_data["injected_context"]
    
    # Add navigation info
    if include_navigation:
        result["navigation"] = _get_navigation_info(chunk_data)
    
    return result


def _get_navigation_info(chunk_data) -> dict:
    """Get navigation info (previous/next chunk IDs) for a chunk with boundary status."""
    with get_chunk_connection() as con:
        cur = con.cursor()
        
        context_id = chunk_data["context_id"]
        sequence = chunk_data["sequence"]
        
        # Find previous chunk in same context, regardless of type
        prev_row = cur.execute(
            """SELECT c.id, c.sequence, c.description 
               FROM chunks c 
               WHERE c.context_id = ? AND c.sequence < ?
               ORDER BY c.sequence DESC LIMIT 1""",
            (context_id, sequence)
        ).fetchone()
        
        # Find next chunk in same context, regardless of type
        next_row = cur.execute(
            """SELECT c.id, c.sequence, c.description 
               FROM chunks c 
               WHERE c.context_id = ? AND c.sequence > ?
               ORDER BY c.sequence ASC LIMIT 1""",
            (context_id, sequence)
        ).fetchone()
        
        # Check if this is the first or last chunk in the file
        is_first_chunk = prev_row is None
        is_last_chunk = next_row is None
        
        # Get total chunks in this context for additional context (regardless of type)
        total_chunks = cur.execute(
            """SELECT COUNT(*) FROM chunks c 
               WHERE c.context_id = ?""",
            (context_id,)
        ).fetchone()[0]
        
        return {
            "previous": {
                "chunk_id": prev_row["id"], 
                "sequence": prev_row["sequence"], 
                "description": prev_row["description"]
            } if prev_row else None,
            "next": {
                "chunk_id": next_row["id"], 
                "sequence": next_row["sequence"], 
                "description": next_row["description"]
            } if next_row else None,
            "boundary_status": {
                "is_first_chunk": is_first_chunk,
                "is_last_chunk": is_last_chunk,
                "total_chunks": total_chunks,
                "current_position": sequence + 1  # 1-based for user display
            }
        }





if __name__ == "__main__":
    try:
        logging.info("Starting HALCON MCP server …")
        validate_database()
        
        # Pre-load models and indices to prevent Claude Desktop timeouts
        logging.info("Warming up semantic indices...")
        _ensure_semantic_index()
        _ensure_full_chunk_index()
        _ensure_micro_chunk_index()
        logging.info("Warmup complete")
        
        logging.info("Launching FastMCP (transport=stdio)…")
        
        # Run with error handling
        mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logging.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logging.exception("Fatal error in HALCON MCP server: %s", e)
        sys.exit(1) 