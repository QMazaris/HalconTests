#!/usr/bin/env python3
"""chunk_scanner_cli.py

CLI utility to parse documentation-like comment blocks from text files and
optionally store them in an SQLite database.

Specification implemented according to request:
 1. Recursively scan files beneath a root directory (default: current dir).
 2. Parse each file into chunks separated by lines matching the pattern
        ^\s*\*\s*(.+)
    Each chunk contains ``title``, ``description`` and ``code`` sections.
 3. DEV mode (default): print chunks to stdout for inspection.
 4. PROD mode: when the --sqlite flag is provided, write chunks into the
    SQLite table ``examples`` instead of printing them.

Only the Python standard-library is used (argparse, os, re, fnmatch, sqlite3).
"""
# Standard library imports only
import argparse
import os
import re
import sqlite3
import hashlib
from fnmatch import fnmatch
from typing import Dict, List, Generator, Optional, Tuple, Set

HEADER_RE = re.compile(r"^\s*\*\s*(.+)")
# Remove XML/HTML-style tags (anything between '<' and '>').
TAG_RE = re.compile(r"<[^>]+>")
DEFAULT_EXTENSIONS = ["hdev", "txt", "halcon"]
# New: reasonable default maximum number of code lines before a chunk is split
DEFAULT_MAX_CODE_LINES = 300
TABLE_DEFINITION = (
    "CREATE TABLE IF NOT EXISTS examples ("
    "  id INTEGER PRIMARY KEY,"
    "  title TEXT NOT NULL,"
    "  description TEXT NOT NULL,"
    "  code TEXT NOT NULL,"
    "  tags TEXT)"
)

# Global set to track seen content hashes for deduplication
SEEN_CONTENT_HASHES: Set[str] = set()
# Counter for tracking duplicates
DUPLICATE_COUNT = 0


############################################################
# Helper functions
############################################################

def is_text_file(path: str, peek_size: int = 1024) -> bool:
    """Return True if *path* appears to contain text data.

    A very lightweight heuristic: read up to *peek_size* bytes and check
    for NUL bytes which are very unlikely in text files. If the file cannot
    be read (e.g., permission issues) treat it as non-text.
    """
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(peek_size)
        return b"\0" not in chunk
    except (OSError, IOError):
        return False


def scan_files(root: str, patterns: List[str]) -> Generator[str, None, None]:
    """Yield file paths under *root* whose basename matches any of *patterns*.

    *patterns* are shell style (fnmatch) patterns such as "*.hdev".
    Only files that also pass the *is_text_file* heuristic are yielded.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if any(fnmatch(fname.lower(), pat) for pat in patterns):
                path = os.path.join(dirpath, fname)
                if is_text_file(path):
                    yield path


def create_content_hash(title: str, description: str, code: str) -> str:
    """Create a hash of the content to identify duplicates"""
    content = f"{title.strip()}\n{description.strip()}\n{code.strip()}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def is_duplicate_content(title: str, description: str, code: str) -> bool:
    """Check if this content is a duplicate of something already seen"""
    global DUPLICATE_COUNT
    content_hash = create_content_hash(title, description, code)
    
    if content_hash in SEEN_CONTENT_HASHES:
        DUPLICATE_COUNT += 1
        return True
    
    SEEN_CONTENT_HASHES.add(content_hash)
    return False


def chunk_file(path: str, max_code_lines: int = DEFAULT_MAX_CODE_LINES) -> List[Dict[str, str]]:
    """Parse *path* into a list of chunk dictionaries.

    Each dictionary has the keys: title, description, code, tags.
    For HALCON files, we treat each meaningful comment block as a section header
    and collect subsequent code lines until the next comment or end of file.

    If *max_code_lines* is provided (>0) and the accumulated *code* block exceeds
    that length, the block is split into multiple chunks, each with at most
    *max_code_lines* lines. The chunk titles receive a "(part N)" suffix to
    keep them unique and readable.
    
    Now includes deduplication to avoid collecting identical content.
    """
    chunks: List[Dict[str, str]] = []
    current_title: Optional[str] = None
    description_lines: List[str] = []
    code_lines: List[str] = []

    def finalize_chunk(title: str, desc_lines: List[str], code_lines_chunk: List[str], tags: str) -> None:
        """Helper to finalize a chunk with deduplication check"""
        description = "\n".join(desc_lines).strip()
        code = "\n".join(code_lines_chunk).strip()
        
        # Skip if duplicate content
        if is_duplicate_content(title, description, code):
            return
        
        chunks.append({
            "title": title.strip(),
            "description": description,
            "code": code,
            "tags": tags
        })

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            # Strip XML/HTML-style tags such as <c>, <l>, </procedure>, etc.
            sanitized_line = TAG_RE.sub("", line).strip()
            
            # Candidate section header line
            if line.strip().startswith('<c>*') and sanitized_line.startswith('*'):
                title_content = sanitized_line.lstrip('*').strip()
                # Skip titles that are just a run of punctuation/dashes
                if re.fullmatch(r"[-\s]{3,}", title_content):
                    i += 1
                    continue
                
                # Finalize previous chunk if any
                if current_title is not None and (description_lines or code_lines):
                    # Apply splitting if code exceeds max_code_lines
                    if max_code_lines and len(code_lines) > max_code_lines > 0:
                        for part_idx in range(0, len(code_lines), max_code_lines):
                            part_lines = code_lines[part_idx: part_idx + max_code_lines]
                            part_title = f"{current_title.strip()} (part {part_idx // max_code_lines + 1})"
                            tags = os.path.splitext(path)[1].lstrip(".")
                            finalize_chunk(part_title, description_lines, part_lines, tags)
                    else:
                        tags = os.path.splitext(path)[1].lstrip(".")
                        finalize_chunk(current_title, description_lines, code_lines, tags)
                
                # Start new chunk using valid title_content
                # Skip empty or very short titles
                if len(title_content) < 3:
                    i += 1
                    continue
                
                current_title = title_content
                description_lines = []
                code_lines = []
                
                # Look ahead for additional description lines
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    next_sanitized = TAG_RE.sub("", next_line).strip()
                    
                    # If it's another comment line that's not a new title, add to description
                    if next_line.strip().startswith('<c>*') and next_sanitized.startswith('*'):
                        desc_content = next_sanitized.lstrip('*').strip()
                        if desc_content and not desc_content.endswith(':'):  # Avoid obvious new section headers
                            description_lines.append(desc_content)
                            j += 1
                        else:
                            break
                    else:
                        break
                
                i = j
                continue
            
            # If we have a current chunk and this is a code line
            elif current_title is not None and line.strip().startswith('<l>'):
                code_lines.append(sanitized_line)
            
            i += 1

        # Finalize last chunk
        if current_title is not None and (description_lines or code_lines):
            if max_code_lines and len(code_lines) > max_code_lines > 0:
                for part_idx in range(0, len(code_lines), max_code_lines):
                    part_lines = code_lines[part_idx: part_idx + max_code_lines]
                    part_title = f"{current_title.strip()} (part {part_idx // max_code_lines + 1})"
                    tags = os.path.splitext(path)[1].lstrip(".")
                    finalize_chunk(part_title, description_lines, part_lines, tags)
            else:
                tags = os.path.splitext(path)[1].lstrip(".")
                finalize_chunk(current_title, description_lines, code_lines, tags)
            
    except (OSError, IOError) as exc:
        print(f"[ERROR] Unable to read {path}: {exc}")

    return chunks


def print_chunks(path: str, chunks: List[Dict[str, str]], verbose: bool = False) -> None:
    for idx, ch in enumerate(chunks, 1):
        if not ch["title"] or not ch["code"]:
            print(f"⚠️  Missing title/code in chunk #{idx} of file {path}")
        header = f"[FILE: {path}] [CHUNK #{idx}] {ch['title']}"
        print(header)
        print("DESC:")
        print(ch["description"])
        print("CODE:")
        print(ch["code"])
        if verbose:
            print("—" * 40)


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(TABLE_DEFINITION)
    return conn


def insert_chunks(conn: sqlite3.Connection, chunks: List[Dict[str, str]]) -> None:
    cur = conn.cursor()
    try:
        cur.execute("BEGIN")
        for ch in chunks:
            if not ch["title"] or not ch["code"]:
                print(f"⚠️  Missing title/code – skipping insertion for chunk '{ch['title']}'")
                continue
            cur.execute(
                "INSERT INTO examples (title, description, code, tags) VALUES (?, ?, ?, ?)",
                (ch["title"], ch["description"], ch["code"], ch["tags"]),
            )
        conn.commit()
    except sqlite3.DatabaseError as exc:
        conn.rollback()
        print(f"[DB-ERROR] {exc}. Transaction has been rolled back.")
    finally:
        cur.close()


############################################################
# Main CLI entry-point
############################################################

def main() -> None:
    # Hardcoded values for direct execution
    root = os.path.abspath("C:\\Users\\QuinnMazaris\\AppData\\Roaming\\MVTec\\HALCON-24.11-Progress-Student-Edition\\examples\\hdevelop")
    sqlite_db_path = "halcon_code_examplesV2.db"
    patterns = [f"*.{ext.strip().lower()}" for ext in DEFAULT_EXTENSIONS if ext.strip()]
    max_code_lines = DEFAULT_MAX_CODE_LINES
    verbose = True  # Set to False to reduce output

    # Clear existing database to start fresh
    if os.path.exists(sqlite_db_path):
        os.remove(sqlite_db_path)
        print(f"Removed existing database: {sqlite_db_path}")

    if verbose:
        print(f"Scanning root: {root}")
        print(f"Using patterns: {patterns}")
        print("Deduplication enabled - identical content will be skipped")

    file_count = 0
    total_chunks: List[Tuple[str, List[Dict[str, str]]]] = []

    for path in scan_files(root, patterns):
        rel_path = os.path.relpath(path, root)
        chunks = chunk_file(path, max_code_lines=max_code_lines)
        total_chunks.append((rel_path, chunks))
        file_count += 1

    if sqlite_db_path:
        # PROD mode: write to database
        conn = init_db(sqlite_db_path)
        for _rel, chunks in total_chunks:
            insert_chunks(conn, chunks)
        conn.close()
        if verbose:
            print(f"Inserted chunks into SQLite database '{sqlite_db_path}'.")
    else:
        # DEV mode: pretty-print
        for rel, chunks in total_chunks:
            print_chunks(rel, chunks, verbose=verbose)

    if verbose:
        chunk_total = sum(len(ch) for _path, ch in total_chunks)
        print(f"Processed {file_count} files – extracted {chunk_total} unique chunks.")
        print(f"Skipped {DUPLICATE_COUNT} duplicate chunks.")
        print(f"Deduplication saved ~{DUPLICATE_COUNT/chunk_total*100:.1f}% space reduction.")


if __name__ == "__main__":
    main() 