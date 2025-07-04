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

from mcp.server.fastmcp import FastMCP

# Configuration Constants
DB_PATH = Path(os.getenv("HALCON_DB_PATH", Path(__file__).with_name("combined.db")))
CODE_DB_PATH = Path(os.getenv("HALCON_CODE_DB_PATH", Path(__file__).with_name("halcon_code_examplesV2.db")))
CHUNK_DB_PATH = Path(os.getenv("HALCON_CHUNK_DB_PATH", Path(__file__).with_name("halcon_chunks_latest.db")))

# Pre-built index paths
OPERATOR_INDEX_PATH = Path(__file__).with_name("halcon_operators.faiss")
OPERATOR_META_PATH = Path(__file__).with_name("halcon_operators_meta.pkl")
CODE_INDEX_PATH = Path(__file__).with_name("halcon_code_examples.faiss")
CODE_META_PATH = Path(__file__).with_name("halcon_code_examples_meta.pkl")
CHUNK_INDEX_PATH = Path(__file__).with_name("halcon_chunks.faiss")
CHUNK_META_PATH = Path(__file__).with_name("halcon_chunks_meta.pkl")

# Semantic search configuration  
SEMANTIC_MODEL_NAME = os.getenv("HALCON_EMBED_MODEL", "microsoft/codebert-base")
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
_chunk_index = None
_chunk_meta = None


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


def _embed_text(text: str, is_query: bool = False) -> np.ndarray:
    """Embed text optimized for the selected model."""
    _ensure_embedding_model()
    
    # CodeBERT and most other models don't need special prefixes
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
            # Combine multiple fields for better search
            text_parts = []
            if row["name"]:
                text_parts.append(row["name"])
            if row["description"]:
                text_parts.append(row["description"])
            if row["signature"]:
                text_parts.append(row["signature"])
            
            text = " ".join(text_parts) if text_parts else "No description available"
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

    # Compute embeddings in batches for better performance
    logging.info("Computing embeddings for %d operators...", len(texts))
    embeddings = _embedding_model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
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
            code_snippet = (row["code"] or "")[:400]  # Increased for better context
            text_parts = []
            if row["title"]:
                text_parts.append(row["title"])
            if row["description"]:
                text_parts.append(row["description"])
            if code_snippet:
                text_parts.append(code_snippet)
            if row["tags"]:
                text_parts.append(row["tags"])
            
            text = " ".join(text_parts).strip() if text_parts else "No content available"
            texts.append(text)
            _code_meta.append(
                {
                    "title": row["title"],
                    "description": row["description"],
                    "code": row["code"],
                    "tags": row["tags"],
                }
            )

    # Compute embeddings in batches for better performance
    logging.info("Computing embeddings for %d code examples...", len(texts))
    embeddings = _embedding_model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    
    # Normalize to unit length for cosine similarity
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


def _ensure_chunk_index() -> None:
    """Load or build the FAISS index with embeddings of code chunks."""
    global _chunk_index, _chunk_meta

    if _chunk_index is not None:
        return  # Already loaded

    # Check if chunk database exists
    if not CHUNK_DB_PATH.exists():
        logging.warning("Chunk database not found - chunk search unavailable")
        return

    # Try to load pre-built index first
    if CHUNK_INDEX_PATH.exists() and CHUNK_META_PATH.exists():
        logging.info("Loading pre-built semantic index for HALCON code chunks...")
        try:
            _chunk_index = faiss.read_index(str(CHUNK_INDEX_PATH))
            with open(CHUNK_META_PATH, 'rb') as f:
                _chunk_meta = pickle.load(f)
            
            _ensure_embedding_model()
            
            # Set nprobe for quantized indices
            if hasattr(_chunk_index, 'nprobe'):
                _chunk_index.nprobe = NPROBE
                
            logging.info("Pre-built chunk index loaded with %d chunk embeddings", len(_chunk_meta))
            return
        except Exception as e:
            logging.warning("Failed to load pre-built chunk index, rebuilding: %s", e)

    # Build index from scratch
    logging.info("Building semantic embedding index for HALCON code chunks...")
    _ensure_embedding_model()

    # Fetch chunk data with context information
    with get_chunk_connection() as con:
        cur = con.cursor()
        
        # Join chunks with contexts to get file information
        rows = cur.execute("""
            SELECT 
                c.id as chunk_id,
                c.context_id, 
                c.chunk_type, 
                c.sequence, 
                c.description, 
                c.code, 
                c.line_start, 
                c.line_end, 
                c.injected_context,
                ctx.file,
                ctx.procedure,
                ctx.header,
                ctx.tags
            FROM chunks c
            JOIN contexts ctx ON c.context_id = ctx.id
            ORDER BY c.context_id, c.sequence
        """).fetchall()

        if not rows:
            logging.warning("No chunks found in database")
            return

        texts: list[str] = []
        _chunk_meta = []

        for row in rows:
            # Combine fields for embedding - prioritize code but include context
            text_parts = []
            
            # Add file context first
            if row["header"]:
                text_parts.append(f"File: {row['header']}")
            if row["procedure"]:
                text_parts.append(f"Procedure: {row['procedure']}")
                
            # Add chunk description
            if row["description"]:
                text_parts.append(f"Description: {row['description']}")
                
            # Add the main code content
            if row["code"]:
                text_parts.append(f"Code: {row['code']}")
                
            # Add injected context if available (but truncated)
            if row["injected_context"]:
                context_preview = row["injected_context"][:200]
                text_parts.append(f"Context: {context_preview}")
                
            # Add tags for additional context
            if row["tags"]:
                text_parts.append(f"Tags: {row['tags']}")
            
            text = " ".join(text_parts).strip() if text_parts else "No content available"
            texts.append(text)
            
            _chunk_meta.append({
                "chunk_id": row["chunk_id"],
                "context_id": row["context_id"],
                "chunk_type": row["chunk_type"],
                "sequence": row["sequence"],
                "description": row["description"],
                "code": row["code"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "injected_context": row["injected_context"],
                "file": row["file"],
                "procedure": row["procedure"],
                "header": row["header"],
                "tags": row["tags"]
            })

    # Compute embeddings in batches for better performance
    logging.info("Computing embeddings for %d chunks in batches...", len(texts))
    embeddings = _embedding_model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )
    
    # Normalize to unit length for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    _chunk_index = _build_faiss_index(embeddings, USE_QUANTIZATION)
    logging.info("Semantic index built with %d chunk embeddings", len(texts))

    # Save index for future runs
    try:
        logging.info("Saving semantic index for code chunks to %s", CHUNK_INDEX_PATH)
        faiss.write_index(_chunk_index, str(CHUNK_INDEX_PATH))
        with open(CHUNK_META_PATH, 'wb') as f:
            pickle.dump(_chunk_meta, f)
        logging.info("Code chunk index saved successfully.")
    except Exception as e:
        logging.warning("Could not save code chunk index: %s", e)


@mcp.resource("halcon://operators")
def get_operators_info() -> str:
    """Get a summary of the HALCON knowledge base, including operator, code example, and chunk counts."""
    with get_connection() as con:
        op_count = con.execute("SELECT COUNT(*) FROM operators").fetchone()[0]

    code_count = 0
    try:
        if CODE_DB_PATH.exists():
            with get_code_connection() as code_con:
                code_count = code_con.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
    except Exception:
        pass  # It's okay if code examples are not available.

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
        f"HALCON knowledge base ready. Contains {op_count} operators, {code_count} code examples{chunk_info}. "
        "Use tools for semantic search, detailed operator lookup, and enhanced chunk search with context injection."
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
        Paginated list of operators.
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
    fields: list[str] = ["name"],
) -> list[dict]:
    """Return top-k semantically matching HALCON operators for a natural language query. Recommended to use default fields.

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
               Default: ["name"]

    Returns:
        List of dictionaries with requested operator information and similarity score.
    """
    try:
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

        # Embed query with proper prefix for E5 model
        vec = _embed_text(query, is_query=True)

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
    
    except Exception as e:
        logging.exception("Error in semantic_match: %s", e)
        return []


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
    try:
        if not query or len(query.strip()) < 3:
            return []

        _ensure_code_index()

        # Embed query with proper prefix for E5 model
        vec = _embed_text(query, is_query=True)

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
    
    except Exception as e:
        logging.exception("Error in semantic_code_search: %s", e)
        return []


@mcp.tool()
def enhanced_chunk_search(
    query: str,
    k: int = TOP_K_DEFAULT,
    chunk_type: str = "all",  # "full", "micro", "all"
    include_context: bool = True,
    navigation: str = "none"  # "none", "prev", "next", "both"
) -> list[dict]:
    """Enhanced semantic search over HALCON code chunks with context injection and navigation.

    This is the unified search function that provides:
    - Search over intelligently chunked HALCON code
    - Context injection for better understanding
    - Navigation to related chunks (previous/next)
    - Choice between full file chunks or micro chunks
    
    Args:
        query: Natural language search string for finding relevant code chunks
        k: Number of matches to return (default 5)
        chunk_type: Type of chunks to search - "full" (complete files), "micro" (code segments), or "all"
        include_context: Whether to include injected context in results for better understanding
        navigation: Include navigation info - "none", "prev" (previous chunk), "next" (next chunk), "both"
        
    Returns:
        List of dictionaries with chunk information, code, context, and navigation data.
        Each result includes: chunk_id, type, description, code, file info, line numbers, 
        and optionally injected_context and navigation links.
    """
    try:
        if not query or len(query.strip()) < 3:
            return []

        _ensure_chunk_index()
        
        if _chunk_index is None or _chunk_meta is None:
            return [{"error": "Chunk database not available. Please ensure chunks have been generated."}]

        # Embed query with proper prefix for E5 model
        vec = _embed_text(query, is_query=True)

        # Filter by chunk type if specified
        valid_indices = []
        if chunk_type == "all":
            valid_indices = list(range(len(_chunk_meta)))
        else:
            for i, meta in enumerate(_chunk_meta):
                if meta["chunk_type"] == chunk_type:
                    valid_indices.append(i)
        
        if not valid_indices:
            return [{"error": f"No chunks found of type '{chunk_type}'"}]

        # Search in the full index but filter results
        D, I = _chunk_index.search(vec.astype(np.float32), min(k * 3, len(_chunk_meta)))

        results = []
        added_count = 0
        
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or added_count >= k:
                break
                
            # Skip if this index is not in our valid set
            if idx not in valid_indices:
                continue
                
            meta = _chunk_meta[idx]
            
            # Build the base result
            result = {
                "chunk_id": meta["chunk_id"],
                "context_id": meta["context_id"],
                "chunk_type": meta["chunk_type"],
                "sequence": meta["sequence"],
                "score": float(score),
                "file": meta["file"],
                "procedure": meta["procedure"],
                "description": meta["description"],
                "code": meta["code"],
                "line_start": meta["line_start"],
                "line_end": meta["line_end"],
                "file_header": meta["header"],
                "tags": meta["tags"]
            }
            
            # Add injected context if requested and available
            if include_context and meta["injected_context"]:
                result["injected_context"] = meta["injected_context"]
            
            # Add navigation information if requested
            if navigation in ["prev", "next", "both"]:
                nav_info = {}
                
                # Find previous and next chunks in the same context
                current_context = meta["context_id"]
                current_sequence = meta["sequence"]
                
                if navigation in ["prev", "both"]:
                    # Find previous chunk
                    prev_chunk = None
                    for other_meta in _chunk_meta:
                        if (other_meta["context_id"] == current_context and 
                            other_meta["sequence"] == current_sequence - 1):
                            prev_chunk = {
                                "chunk_id": other_meta["chunk_id"],
                                "sequence": other_meta["sequence"],
                                "description": other_meta["description"]
                            }
                            break
                    nav_info["previous"] = prev_chunk
                
                if navigation in ["next", "both"]:
                    # Find next chunk
                    next_chunk = None
                    for other_meta in _chunk_meta:
                        if (other_meta["context_id"] == current_context and 
                            other_meta["sequence"] == current_sequence + 1):
                            next_chunk = {
                                "chunk_id": other_meta["chunk_id"],
                                "sequence": other_meta["sequence"],
                                "description": other_meta["description"]
                            }
                            break
                    nav_info["next"] = next_chunk
                
                result["navigation"] = nav_info
            
            results.append(result)
            added_count += 1

        return results
    
    except Exception as e:
        logging.exception("Error in enhanced_chunk_search: %s", e)
        return [{"error": f"Search failed: {str(e)}"}]


if __name__ == "__main__":
    try:
        logging.info("Starting HALCON MCP server …")
        validate_database()
        
        # Pre-load models and indices to prevent Claude Desktop timeouts
        logging.info("Warming up semantic indices...")
        _ensure_semantic_index()
        _ensure_code_index()
        _ensure_chunk_index()
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