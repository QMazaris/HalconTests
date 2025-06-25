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

# File paths
SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "combined.db"
CODE_DB_PATH = SCRIPT_DIR / "halcon_code_examplesV2.db"

# Output paths
OPERATOR_INDEX_PATH = SCRIPT_DIR / "halcon_operators.faiss"
OPERATOR_META_PATH = SCRIPT_DIR / "halcon_operators_meta.pkl"
CODE_INDEX_PATH = SCRIPT_DIR / "halcon_code_examples.faiss"
CODE_META_PATH = SCRIPT_DIR / "halcon_code_examples_meta.pkl"


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


def build_operator_index():
    """Build and save the operator semantic search index."""
    logging.info("Building HALCON operators semantic index...")
    
    if not DB_PATH.exists():
        logging.error("Operators database not found: %s", DB_PATH)
        return False
    
    # Load embedding model
    embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    
    # Fetch operator data
    con = get_connection()
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT name, description, signature, url FROM operators"
        ).fetchall()

        texts: List[str] = []
        operator_meta: List[Dict] = []

        for row in rows:
            text = row["description"] or "No description available"
            texts.append(text)
            operator_meta.append({
                "name": row["name"],
                "description": row["description"] or "No description available",
                "signature": row["signature"],
                "url": row["url"],
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

    dim = embeddings.shape[1]
    n_vectors = len(embeddings)
    
    logging.info("Creating FAISS index with %d vectors of dimension %d", n_vectors, dim)

    # Create optimized index
    if USE_QUANTIZATION and n_vectors > 1000:
        # Use IVF (Inverted File) with inner product for cosine similarity on normalized vectors
        n_centroids = min(max(int(np.sqrt(n_vectors)), 100), n_vectors // 10)
        faiss_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
        logging.info("Training quantized index with %d centroids...", n_centroids)
        faiss_index.train(embeddings.astype(np.float32))
        faiss_index.add(embeddings.astype(np.float32))
        faiss_index.nprobe = NPROBE
        logging.info("Built quantized IVF index")
    else:
        # Use flat index for small datasets or exact search
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings.astype(np.float32))
        logging.info("Built flat index for exact search")

    # Save index and metadata
    logging.info("Saving index to %s", OPERATOR_INDEX_PATH)
    faiss.write_index(faiss_index, str(OPERATOR_INDEX_PATH))
    
    logging.info("Saving metadata to %s", OPERATOR_META_PATH)
    with open(OPERATOR_META_PATH, 'wb') as f:
        pickle.dump(operator_meta, f)
    
    logging.info("✅ Operator index built successfully")
    return True


def build_code_index():
    """Build and save the code examples semantic search index."""
    logging.info("Building HALCON code examples semantic index...")
    
    if not CODE_DB_PATH.exists():
        logging.warning("Code examples database not found: %s", CODE_DB_PATH)
        logging.info("Skipping code index build")
        return True  # Not a fatal error
    
    # Load embedding model
    embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    
    # Fetch code example data
    con = get_code_con()
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT title, description, code, tags FROM examples"
        ).fetchall()

        texts: List[str] = []
        code_meta: List[Dict] = []

        for row in rows:
            # Combine available textual fields for embedding, excluding code content
            text_parts = [row["title"] or "", row["description"] or ""]
            texts.append(" ".join(text_parts))
            code_meta.append({
                "title": row["title"],
                "description": row["description"],
                "code": row["code"],
                "tags": row["tags"],
            })
    finally:
        con.close()

    if not texts:
        logging.warning("No code examples found, skipping code index build")
        return True

    logging.info("Loaded %d code examples for embedding", len(texts))

    # Compute embeddings
    logging.info("Computing code embeddings...")
    embeddings = embedding_model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dim = embeddings.shape[1]
    n_vectors = len(embeddings)
    
    logging.info("Creating code FAISS index with %d vectors of dimension %d", n_vectors, dim)

    # Create optimized index
    if USE_QUANTIZATION and n_vectors > 500:
        # Use IVF for code examples (smaller dataset, fewer centroids)
        n_centroids = min(max(int(np.sqrt(n_vectors)), 50), n_vectors // 5)
        faiss_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
        logging.info("Training quantized code index with %d centroids...", n_centroids)
        faiss_index.train(embeddings.astype(np.float32))
        faiss_index.add(embeddings.astype(np.float32))
        faiss_index.nprobe = NPROBE
        logging.info("Built quantized code index")
    else:
        # Use flat index for small datasets
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings.astype(np.float32))
        logging.info("Built flat code index for exact search")

    # Save index and metadata
    logging.info("Saving code index to %s", CODE_INDEX_PATH)
    faiss.write_index(faiss_index, str(CODE_INDEX_PATH))
    
    logging.info("Saving code metadata to %s", CODE_META_PATH)
    with open(CODE_META_PATH, 'wb') as f:
        pickle.dump(code_meta, f)
    
    logging.info("✅ Code index built successfully")
    return True


def main():
    """Build all semantic search indices."""
    logging.info("🚀 Building pre-computed semantic search indices...")
    logging.info("Using embedding model: %s", SEMANTIC_MODEL_NAME)
    logging.info("Quantization enabled: %s", USE_QUANTIZATION)
    
    success = True
    
    # Build operator index
    if not build_operator_index():
        success = False
    
    # Build code index
    if not build_code_index():
        success = False
    
    if success:
        logging.info("🎉 All indices built successfully!")
        logging.info("\nGenerated files:")
        for path in [OPERATOR_INDEX_PATH, OPERATOR_META_PATH, CODE_INDEX_PATH, CODE_META_PATH]:
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