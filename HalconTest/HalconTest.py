import sqlite3
import os
import pickle
from pathlib import Path
import sys
from typing import Optional, Union
import logging
import re
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

# How many extra candidates to pull from FAISS before filtering/deduping.
# e.g. factor 5 means we fetch up to k*5 results per index.
EXTRA_RESULTS_FACTOR = 10

# Boost that is added to the semantic similarity score if the exact query
# token is found as a whole word inside the code snippet. Helps single-operator
# queries (e.g. "read_dl_model") surface direct hits.
KEYWORD_BOOST = 5.0

# Cached set of all HALCON operator names (lower-case)
_operator_names: set[str] | None = None


def _ensure_operator_name_set() -> set[str]:
    """Load and cache the set of HALCON operator names from the operators table."""
    global _operator_names
    if _operator_names is None:
        with get_connection() as con:
            rows = con.execute("SELECT name FROM operators").fetchall()
        _operator_names = {r[0].lower() for r in rows}
    return _operator_names

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
    """Return a one-line summary of the loaded HALCON knowledge base (operator & chunk counts)."""
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
    """HALCON operator search. Single-token ⇒ exact match, multi-token ⇒ semantic search.
    
    Parameters:
    - query: Operator name or description (min 2 chars)
    - fields: Data to return ["name", "signature", "description", "parameters", "results", "url"] or "all"
    - k: Max results for semantic search (ignored for exact matches)
    
    Returns dict (exact match), list[dict] (semantic results), or error string.
    """
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


def _dedupe_adjacent_micro_results(results: list[dict]) -> list[dict]:
    """Remove near-duplicate micro-chunk hits that originate from adjacent
    parts of the same file.

    If two micro chunks come from the same ``context_id`` (i.e. file) and their
    ``sequence`` numbers differ by 1 (direct neighbours), only the first (highest
    scored) occurrence will be kept. This reduces visual clutter in the top-k
    results while still allowing navigation via the *next/previous* links.
    """
    deduped: list[dict] = []
    # Map context_id -> list of already kept sequence numbers
    seen_sequences: dict[int, list[int]] = {}

    for res in results:
        # Only dedupe micro chunks; keep other chunk types untouched
        if res.get("chunk_type") != "micro":
            deduped.append(res)
            continue

        ctx_id = res.get("context_id")
        seq = res.get("sequence")
        if ctx_id is None or seq is None:
            # Should not happen, but keep defensive
            deduped.append(res)
            continue

        # Skip if a neighbouring sequence (±1) from the same context was kept
        if ctx_id in seen_sequences and any(abs(seq - s) <= 1 for s in seen_sequences[ctx_id]):
            continue  # near-duplicate → drop

        deduped.append(res)
        seen_sequences.setdefault(ctx_id, []).append(seq)

    return deduped


# --- Exact token fallback ----------------------------------------------------

def _exact_match_chunks(token: str, chunk_type_filter: str, include_context: bool,
                       include_navigation: bool, limit: int = 10) -> list[dict]:
    """Return chunks whose *code* field contains the exact token (case-insensitive).

    For operator-like single-word queries we run a cheap SQL LIKE search
    directly on the SQLite chunks table. This acts as a safety net when the
    semantic model fails to surface a relevant snippet.
    """
    if not CHUNK_DB_PATH.exists():
        return []

    token_like = f"%{token}%"
    rows: list[sqlite3.Row] = []
    with get_chunk_connection() as con:
        cur = con.cursor()

        if chunk_type_filter in ("full", "micro"):
            cur.execute(
                """SELECT c.id as chunk_id, c.context_id, c.chunk_type, c.sequence, c.description, c.code,
                          c.line_start, c.line_end, c.injected_context,
                          ctx.file, ctx.procedure, ctx.header, ctx.tags
                   FROM chunks c JOIN contexts ctx ON c.context_id = ctx.id
                   WHERE c.chunk_type = ? AND LOWER(c.code) LIKE LOWER(?)
                   LIMIT ?""",
                (chunk_type_filter, token_like, limit),
            )
        else:  # 'all'
            cur.execute(
                """SELECT c.id as chunk_id, c.context_id, c.chunk_type, c.sequence, c.description, c.code,
                          c.line_start, c.line_end, c.injected_context,
                          ctx.file, ctx.procedure, ctx.header, ctx.tags
                   FROM chunks c JOIN contexts ctx ON c.context_id = ctx.id
                   WHERE LOWER(c.code) LIKE LOWER(?)
                   LIMIT ?""",
                (token_like, limit),
            )
        rows = cur.fetchall()

    # Build results with a high artificial score so they bubble to the top
    results = [
        _build_chunk_result(r, r["chunk_type"], KEYWORD_BOOST + 50.0, include_context, include_navigation)
        for r in rows
    ]
    for r in results:
        r["exact_match"] = True
    return results


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
    """Semantic code search & chunk navigation.

    Query mode: Embeds query → searches FAISS indices → returns ranked code chunks.
    - chunk_type: "full" (complete procedures), "micro" (small snippets), or "all"
    - Auto-detects HALCON operators in query for enhanced filtering
    - include_context: Injects surrounding code for micro chunks

    Navigation mode: Direct chunk retrieval by ID.
    - chunk_id: Specific chunk to fetch
    - direction: "previous"/"next" to get adjacent chunk in same file
    - Always includes navigation metadata for browsing between chunks
    """
    try:
        # Handle direct chunk ID lookup (navigation mode)
        if chunk_id is not None:
            return _get_chunk_by_id(chunk_id, direction, chunk_type, include_context, include_navigation)

        # Handle semantic search
        if not query or len(query.strip()) < 3:
            return []

        query = query.strip()
        # Log the incoming search query and parameters
        logging.info("[search_code] query='%s', chunk_type=%s, k=%d", query, chunk_type, k)

        tokens = re.findall(r"\w+", query.lower())
        operator_set = _ensure_operator_name_set()
        operator_tokens = [t for t in tokens if t in operator_set]
        has_operator_in_query = bool(operator_tokens)

        if has_operator_in_query:
            logging.info("[search_code] Detected operator tokens: %s", ", ".join(operator_tokens))

        # Ensure relevant indices are loaded
        if chunk_type in ("full", "all"):
            _ensure_full_chunk_index()
        if chunk_type in ("micro", "all"):
            _ensure_micro_chunk_index()

        all_results: list[dict] = []

        # Track statistics
        total_exact_matches: int = 0  # how many results added via SQL fallback
        total_raw_semantic: int = 0   # hits returned directly from FAISS before filtering
        total_filtered_semantic: int = 0  # hits kept after operator filtering

        # Pre-compile patterns that must appear in the code if operator detected
        op_patterns = [re.compile(r"\b" + re.escape(tok) + r"\b", re.IGNORECASE) for tok in operator_tokens]

        # Helper to search a specific index
        def _search_index(index, meta, source_name: str):
            nonlocal total_raw_semantic, total_filtered_semantic

            vec = _embed_text(query, is_query=True)
            fetch_k = min(len(meta), k * EXTRA_RESULTS_FACTOR)
            D, I = index.search(vec.astype(np.float32), fetch_k)

            for idx, score in zip(I[0], D[0]):
                if idx < 0:
                    continue

                total_raw_semantic += 1  # Count every valid FAISS hit

                m = meta[idx]

                # Prepare texts for operator presence check
                code_text = m.get("code", "") if isinstance(m, dict) else m["code"]
                injected_context_text = ""
                if source_name == "micro":
                    injected_context_text = (
                        m.get("injected_context", "") if isinstance(m, dict) else m.get("injected_context", "")
                    ) or ""

                # Determine if we keep based on operator tokens (when present)
                keep = True
                if has_operator_in_query:
                    has_in_code = any(p.search(code_text) for p in op_patterns)
                    has_in_context = (
                        source_name == "micro" and injected_context_text and any(p.search(injected_context_text) for p in op_patterns)
                    )
                    keep = has_in_code or has_in_context

                if not keep:
                    continue  # filter out

                total_filtered_semantic += 1

                result = _build_chunk_result(m, source_name, float(score), include_context, include_navigation)
                all_results.append(result)

        if chunk_type in ("full", "all") and _full_chunk_index is not None:
            _search_index(_full_chunk_index, _full_chunk_meta, "full")
        if chunk_type in ("micro", "all") and _micro_chunk_index is not None:
            _search_index(_micro_chunk_index, _micro_chunk_meta, "micro")

        # Log semantic retrieval statistics before SQL fallback
        logging.info(
            "[search_code] Semantic hits: %d raw, %d after exact-match filtering",
            total_raw_semantic,
            total_filtered_semantic,
        )

        # If operators are in the query and no semantic hit contains any of them,
        # pull chunks directly via SQL LIKE for each operator token.
        if has_operator_in_query and not any(any(p.search(r.get("code", "")) for p in op_patterns) for r in all_results):
            for tok in operator_tokens:
                exact_matches = _exact_match_chunks(tok, chunk_type, include_context, include_navigation, k)
                total_exact_matches += len(exact_matches)
                all_results.extend(exact_matches)

        if total_exact_matches:
            logging.info("[search_code] Exact-match fallback added %d results", total_exact_matches)

        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Dedupe adjacent micro chunks (same context, neighbouring sequence)
        len_before_dedupe = len(all_results)
        all_results = _dedupe_adjacent_micro_results(all_results)
        len_after_dedupe = len(all_results)

        if len_after_dedupe < len_before_dedupe:
            logging.info("[search_code] Deduped adjacent micro chunks: removed %d duplicates", len_before_dedupe - len_after_dedupe)

        # Ensure we return at most k results.
        final_results = all_results[:k]
        logging.info("[search_code] Returning %d results", len(final_results))
        return final_results

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