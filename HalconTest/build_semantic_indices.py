#!/usr/bin/env python3
"""
Build pre-computed semantic search indices for HALCON MCP Server.
This script creates FAISS indices and metadata files that can be distributed
to users to avoid the need to build embeddings on first use.
"""

import os
import pickle
import logging
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import sqlite3
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Configuration (should match HalconTest.py)
SEMANTIC_MODEL_NAME = os.getenv("HALCON_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_QUANTIZATION = True
QUANTIZATION_BITS = 8
NPROBE = 16

# File paths - updated to use databases/ folder
SCRIPT_DIR = Path(__file__).parent
DB_DIR = SCRIPT_DIR / "databases"
# Paths to databases
DB_PATH = DB_DIR / "combined.db"                   # Operators DB (unchanged)
CHUNK_DB_PATH = DB_DIR / "halcon_chunks_latest.db"  # Unified chunks DB (full + micro)

# Output paths (also in databases/ folder)
OPERATOR_INDEX_PATH = DB_DIR / "halcon_operators.faiss"
OPERATOR_META_PATH = DB_DIR / "halcon_operators_meta.pkl"

# Separated chunk indices
FULL_CHUNK_INDEX_PATH = DB_DIR / "halcon_chunks_full.faiss"
FULL_CHUNK_META_PATH  = DB_DIR / "halcon_chunks_full_meta.pkl"
MICRO_CHUNK_INDEX_PATH = DB_DIR / "halcon_chunks_micro.faiss"
MICRO_CHUNK_META_PATH  = DB_DIR / "halcon_chunks_micro_meta.pkl"


def get_connection():
    """Get database connection."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def get_chunk_connection():
    """Return SQLite connection to the HALCON code chunks database."""
    con = sqlite3.connect(CHUNK_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


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


def build_operator_index():
    """Build and save the operator semantic search index."""
    logging.info("Building HALCON operators semantic index...")
    
    if not DB_PATH.exists():
        logging.error("Operators database not found: %s", DB_PATH)
        return False
    
    # Load embedding model
    embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    
    # Fetch operator data - using same logic as HalconTest.py
    con = get_connection()
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT name, description, signature, url, parameters, results FROM operators"
        ).fetchall()

        texts: List[str] = []
        operator_meta: List[Dict] = []

        for row in rows:
            # Combine multiple fields for better search (same as HalconTest.py)
            text_parts = []
            if row["name"]:
                text_parts.append(row["name"])
            if row["description"]:
                text_parts.append(row["description"])
            if row["signature"]:
                text_parts.append(row["signature"])
            
            text = " ".join(text_parts) if text_parts else "No description available"
            texts.append(text)
            operator_meta.append({
                "name": row["name"],
                "description": row["description"] or "No description available",
                "signature": row["signature"],
                "url": row["url"],
                "parameters": row["parameters"],
                "results": row["results"],
            })
    finally:
        con.close()

    logging.info("Loaded %d operators for embedding", len(texts))

    # Compute embeddings
    logging.info("Computing embeddings...")
    embeddings = embedding_model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )

    # Normalize to unit length for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    faiss_index = _build_faiss_index(embeddings, USE_QUANTIZATION)
    logging.info("Semantic index built with %d operator embeddings", len(texts))

    # Save index and metadata
    logging.info("Saving index to %s", OPERATOR_INDEX_PATH)
    faiss.write_index(faiss_index, str(OPERATOR_INDEX_PATH))
    
    logging.info("Saving metadata to %s", OPERATOR_META_PATH)
    with open(OPERATOR_META_PATH, 'wb') as f:
        pickle.dump(operator_meta, f)
    
    logging.info("✅ Operator index built successfully")
    return True


# (Removed outdated build_code_index function)

# -----------------------------------------------------------------------------
# New builders: one generic that filters by chunk_type, and two thin wrappers
# -----------------------------------------------------------------------------

def _build_filtered_chunk_index(chunk_type_filter: str) -> bool:
    """Build a FAISS index for a specific chunk_type ('full' or 'micro')."""
    if not CHUNK_DB_PATH.exists():
        logging.error("Chunk database not found: %s", CHUNK_DB_PATH)
        return False

    # Load embedding model once
    embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

    con = get_chunk_connection()
    try:
        cur = con.cursor()
        rows = cur.execute(
            """SELECT c.id as chunk_id, c.context_id, c.chunk_type, c.sequence, c.description, c.code,
                      c.line_start, c.line_end, c.injected_context, ctx.file, ctx.procedure, ctx.header, ctx.tags
                   FROM chunks c
                   JOIN contexts ctx ON c.context_id = ctx.id
                   WHERE c.chunk_type = ?
                   ORDER BY c.context_id, c.sequence""",
            (chunk_type_filter,)
        ).fetchall()

        if not rows:
            logging.warning("No %s chunks found in database", chunk_type_filter)
            return True  # Not fatal – just skip

        texts: List[str] = []
        metas: List[Dict] = []

        for row in rows:
            text_parts = []
            if row["header"]:
                text_parts.append(f"File: {row['header']}")
            if row["procedure"]:
                text_parts.append(f"Procedure: {row['procedure']}")
            if row["description"]:
                text_parts.append(f"Description: {row['description']}")

            if row["code"]:
                snippet = row["code"] if chunk_type_filter == "micro" else row["code"][:800]
                text_parts.append(f"Code: {snippet}")

            if chunk_type_filter == "micro" and row["injected_context"]:
                context_preview = row["injected_context"][:200]
                text_parts.append(f"Context: {context_preview}")

            if row["tags"]:
                text_parts.append(f"Tags: {row['tags']}")

            texts.append(" ".join(text_parts).strip())
            metas.append({
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
    finally:
        con.close()

    logging.info("Loaded %d %s chunks for embedding", len(texts), chunk_type_filter)

    # Compute embeddings
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    faiss_index = _build_faiss_index(embeddings, USE_QUANTIZATION)

    if chunk_type_filter == "full":
        index_path = FULL_CHUNK_INDEX_PATH
        meta_path = FULL_CHUNK_META_PATH
    else:
        index_path = MICRO_CHUNK_INDEX_PATH
        meta_path = MICRO_CHUNK_META_PATH

    logging.info("Saving %s chunk index to %s", chunk_type_filter.upper(), index_path)
    faiss.write_index(faiss_index, str(index_path))

    with open(meta_path, "wb") as f:
        pickle.dump(metas, f)

    logging.info("✅ %s chunk index built successfully", chunk_type_filter.upper())
    return True


def build_full_chunk_index():
    return _build_filtered_chunk_index("full")


def build_micro_chunk_index():
    return _build_filtered_chunk_index("micro")


def main():
    """Build all semantic search indices."""
    logging.info("🚀 Building pre-computed semantic search indices...")
    logging.info("Using embedding model: %s", SEMANTIC_MODEL_NAME)
    logging.info("Quantization enabled: %s", USE_QUANTIZATION)
    
    success = True
    
    # Build operator index
    if not build_operator_index():
        success = False
    
    # Build full and micro chunk indices
    if not build_full_chunk_index():
        success = False
    if not build_micro_chunk_index():
        success = False
    
    if success:
        logging.info("🎉 All indices built successfully!")
        logging.info("\nGenerated files:")
        for path in [OPERATOR_INDEX_PATH, OPERATOR_META_PATH, FULL_CHUNK_INDEX_PATH, FULL_CHUNK_META_PATH, MICRO_CHUNK_INDEX_PATH, MICRO_CHUNK_META_PATH]:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                logging.info("  %s (%.1f MB)", path.name, size_mb)
        
        logging.info("\nThese files can be distributed with your MCP server")
        logging.info("to avoid rebuilding indices on first use.")
    else:
        logging.error("❌ Some indices failed to build")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 