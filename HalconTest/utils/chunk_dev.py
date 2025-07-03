#!/usr/bin/env python3
"""chunk_scanner_cli_terminal.py

Utility to parse documentation-like comment blocks from text files and
print them to the terminal for inspection.

Modified version that:
 1. Always prints to terminal (no database functionality)
 2. Allows testing with a single file or directory
 3. Maintains the same parsing logic as the original

Only the Python standard-library is used (os, re, fnmatch).
"""

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES TO CONFIGURE THE SCRIPT
# ============================================================================

# File number to process (0-based index from the available files)
# Change this number to select which file you want to process
# The script will first scan all available files, then process the one at this index
# Examples:
#   FILE_NUMBER = 0    # Process the first file found
#   FILE_NUMBER = 1    # Process the second file found
#   FILE_NUMBER = 42   # Process the 43rd file found
FILE_NUMBER = 5

# Script mode: 'development' for single file testing, 'database' for creating full database
MODE = "database"  # Options: "development", "database"

# File extensions to process
EXTENSIONS = ["hdev", "txt", "halcon"]

# Maximum code lines per chunk before splitting
MAX_CODE_LINES = 300

# Maximum number of context lines to inject per chunk (prevents context explosion)
MAX_CONTEXT_LINES = 50

# Enable verbose output with separators between chunks
VERBOSE = False

# HALCON examples directory to scan for files
HALCON_DIR = r"C:\Users\QuinnMazaris\AppData\Roaming\MVTec\HALCON-25.05-Progress\examples\hdevelop"

# Database configuration
DB_NAME_PREFIX = "halcon_chunks"  # Will append timestamp for uniqueness

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Standard library imports only
import os
import re
import hashlib
import sqlite3
import datetime
from fnmatch import fnmatch
from typing import Dict, List, Generator, Optional, Set, Tuple

HEADER_RE = re.compile(r"^\s*\*\s*(.+)")
# Remove XML/HTML-style tags (anything between '<' and '>').
TAG_RE = re.compile(r"<[^>]+>")

# Regex to find HALCON-style variable names (PascalCase)
VAR_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_]*\b")

# Global set to track seen content hashes for deduplication
SEEN_CONTENT_HASHES: Set[str] = set()
# Counter for tracking duplicates
DUPLICATE_COUNT = 0


def clear_global_state():
    """Clear global state between files to prevent cross-contamination"""
    global SEEN_CONTENT_HASHES, DUPLICATE_COUNT
    SEEN_CONTENT_HASHES.clear()
    DUPLICATE_COUNT = 0


############################################################
# Database functions
############################################################

def create_database(db_path: str) -> sqlite3.Connection:
    """
    Create a new SQLite database with contexts and chunks tables.
    
    Returns the database connection.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create contexts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contexts (
            id TEXT PRIMARY KEY,
            file TEXT NOT NULL,
            procedure TEXT,
            header TEXT,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context_id TEXT NOT NULL,
            chunk_type TEXT NOT NULL CHECK (chunk_type IN ('full', 'micro')),
            sequence INTEGER NOT NULL,
            description TEXT,
            code TEXT NOT NULL,
            line_start INTEGER,
            line_end INTEGER,
            injected_context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (context_id) REFERENCES contexts (id)
        )
    """)
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_context_id ON chunks (context_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks (chunk_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_sequence ON chunks (context_id, sequence)")
    
    conn.commit()
    print(f"✅ Database created: {db_path}")
    print(f"✅ Tables created: contexts, chunks")
    print(f"✅ Indexes created for performance")
    
    return conn


def insert_context(conn: sqlite3.Connection, context: Dict[str, str]) -> None:
    """Insert a context record into the database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO contexts (id, file, procedure, header, tags)
        VALUES (?, ?, ?, ?, ?)
    """, (
        context['id'],
        context['file'],
        context['procedure'],
        context['header'],
        context['tags']
    ))
    conn.commit()


def insert_chunk(conn: sqlite3.Connection, chunk: Dict[str, str]) -> None:
    """Insert a chunk record into the database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chunks (context_id, chunk_type, sequence, description, code, 
                          line_start, line_end, injected_context)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chunk['context_id'],
        chunk['chunk_type'],
        chunk['sequence'],
        chunk['description'],
        chunk['code'],
        chunk['line_start'],
        chunk['line_end'],
        chunk['injected_context']
    ))
    conn.commit()


def generate_db_name() -> str:
    """Generate a unique database name with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{DB_NAME_PREFIX}_{timestamp}.db"


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


def get_defs_for_line(line: str, already_defined: Set[str]) -> List[str]:
    """
    Identifies newly defined variables on a given line of code.
    Uses a "first appearance" heuristic.
    """
    defs = []
    
    # `Var := Val` assignment is a clear definition
    match = re.match(r"^\s*([A-Z][a-zA-Z0-9_]*)\s*:=\s*.*", line)
    if match:
        var = match.group(1)
        if var not in already_defined:
            defs.append(var)
        return defs

    # Operator call: new variables are treated as definitions
    variables_on_line = VAR_RE.findall(line)
    for var in variables_on_line:
        if var not in already_defined:
            defs.append(var)
    return defs


def extract_file_context(path: str) -> Dict[str, str]:
    """
    Extract file-level context information from HALCON files to match the contexts table schema.
    
    Returns a dictionary with keys: file, procedure, header, tags
    """
    context = {
        "file": os.path.basename(path),
        "procedure": None,
        "header": "",
        "tags": ""
    }
    
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        
        # Extract procedure name from XML
        procedure_match = re.search(r'<procedure name="([^"]+)">', content)
        if procedure_match:
            context["procedure"] = procedure_match.group(1)
        
        # Extract file-level header from initial comments
        lines = content.split('\n')
        header_lines = []
        in_body = False
        
        for line in lines:
            stripped = line.strip()
            
            # Look for the start of the body section
            if '<body>' in stripped:
                in_body = True
                continue
            
            # Only collect comments that are in the body section
            if in_body and stripped.startswith('<c>*'):
                # Clean up the comment
                clean_comment = TAG_RE.sub("", stripped).strip()
                clean_comment = clean_comment.lstrip('*').strip()
                if clean_comment and not clean_comment.startswith('---'):
                    header_lines.append(clean_comment)
            
            # Stop collecting header when we hit the first code line
            elif in_body and stripped.startswith('<l>'):
                break
        
        context["header"] = " ".join(header_lines).strip()
        
        # Extract tags from file extension and version info
        file_ext = os.path.splitext(path)[1].lstrip(".")
        tags = [file_ext]
        
        # Add halcon version if available
        version_match = re.search(r'halcon_version="([^"]+)"', content)
        if version_match:
            tags.append(f"halcon-{version_match.group(1)}")
        
        # Add file version if available  
        file_version_match = re.search(r'file_version="([^"]+)"', content)
        if file_version_match:
            tags.append(f"file-v{file_version_match.group(1)}")
        
        context["tags"] = ",".join(tags)
        
    except (OSError, IOError) as exc:
        print(f"[WARNING] Unable to extract context from {path}: {exc}")
    
    return context


def create_full_chunk(path: str, context_id: str) -> Dict[str, str]:
    """
    Create a full chunk containing all code from the file.
    
    Returns a dictionary ready for database insertion.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        
        # Extract all code lines with line numbers
        code_lines = []
        line_numbers = []
        
        for i, line in enumerate(lines, 1):
            sanitized_line = TAG_RE.sub("", line).strip()
            if line.strip().startswith('<l>') and sanitized_line:
                code_lines.append(sanitized_line)
                line_numbers.append(i)
        
        # Create description from initial comments
        description_lines = []
        in_body = False
        
        for line in lines:
            stripped = line.strip()
            if '<body>' in stripped:
                in_body = True
                continue
            if in_body and stripped.startswith('<c>*'):
                clean_comment = TAG_RE.sub("", stripped).strip()
                clean_comment = clean_comment.lstrip('*').strip()
                if clean_comment:
                    description_lines.append(clean_comment)
            elif in_body and stripped.startswith('<l>'):
                break
        
        return {
            "context_id": context_id,
            "chunk_type": "full",
            "sequence": 0,  # Full chunk always comes first
            "description": "\n".join(description_lines).strip(),
            "code": "\n".join(code_lines),
            "line_start": line_numbers[0] if line_numbers else 1,
            "line_end": line_numbers[-1] if line_numbers else 1,
            "injected_context": ""  # No injected context for full chunks
        }
        
    except (OSError, IOError) as exc:
        print(f"[ERROR] Unable to create full chunk from {path}: {exc}")
        return {}


def create_micro_chunks(path: str, context_id: str, max_code_lines: int = MAX_CODE_LINES) -> List[Dict[str, str]]:
    """
    Create intelligent micro chunks with context injection.
    
    Returns a list of dictionaries ready for database insertion.
    """
    micro_chunks = []
    current_title: Optional[str] = None
    description_lines: List[str] = []
    code_lines: List[str] = []
    sequence = 1  # Start at 1 since full chunk is sequence 0

    # Stores (code_line, defined_vars_list, line_number) for context
    historical_context: List[Tuple[str, List[str], int]] = []

    def finalize_micro_chunk(title: str, desc_lines: List[str], code_lines_chunk: List[str], 
                           start_line: int, end_line: int) -> None:
        """
        Finalizes a micro chunk with context injection and adds it to the list.
        """
        nonlocal sequence
        
        # Set of variables defined in the code processed *before* this chunk
        vars_defined_in_history = set(d for _, defs, _ in historical_context for d in defs)

        # 1. Find all variables used in this chunk
        all_vars_in_chunk = set()
        for line in code_lines_chunk:
            all_vars_in_chunk.update(VAR_RE.findall(line))

        # 2. Find variables defined within this chunk
        defined_in_chunk = set()
        temp_defined_this_chunk = vars_defined_in_history.copy()
        for line in code_lines_chunk:
            newly_defined = get_defs_for_line(line, temp_defined_this_chunk)
            defined_in_chunk.update(newly_defined)
            temp_defined_this_chunk.update(newly_defined)
        
        # 3. Find variables that require context from previous code
        needed_vars = all_vars_in_chunk - defined_in_chunk
        
        # 4. Build the context code block from history iteratively
        context_code_lines = []
        context_line_set = set()
        vars_to_resolve = needed_vars.copy()
        
        while vars_to_resolve and len(context_code_lines) < MAX_CONTEXT_LINES:
            new_dependencies_found = set()
            
            for line, defs, line_num in historical_context:
                if any(var in vars_to_resolve for var in defs) and line not in context_line_set:
                    context_code_lines.append(line)
                    context_line_set.add(line)
                    
                    if len(context_code_lines) >= MAX_CONTEXT_LINES:
                        break
                    
                    for used_var in VAR_RE.findall(line):
                        new_dependencies_found.add(used_var)

            # Get all variables defined in the context we just gathered
            vars_defined_in_context = set()
            temp_already_defined = set() 
            for line in context_code_lines:
                defs = get_defs_for_line(line, temp_already_defined)
                vars_defined_in_context.update(defs)
                temp_already_defined.update(defs)

            vars_to_resolve = new_dependencies_found - vars_defined_in_context - defined_in_chunk - needed_vars

        # Sort context lines by their original order in the file
        if context_code_lines:
            line_to_position = {}
            for idx, (line, _, line_num) in enumerate(historical_context):
                line_to_position[line] = line_num
            context_code_lines.sort(key=lambda line: line_to_position.get(line, float('inf')))

        # 5. Prepare injected context
        injected_context = ""
        if context_code_lines:
            context_header = ["* --- Automatically Added Context ---"]
            if len(context_code_lines) >= MAX_CONTEXT_LINES:
                context_header.append(f"* WARNING: Context truncated at {MAX_CONTEXT_LINES} lines")
            context_footer = ["* --- End of Automatically Added Context ---"]
            injected_context = "\n".join(context_header + context_code_lines + context_footer)

        description = "\n".join(desc_lines).strip()
        code = "\n".join(code_lines_chunk).strip()
        
        if not code:
            return

        if is_duplicate_content(title, description, code):
            return
        
        micro_chunks.append({
            "context_id": context_id,
            "chunk_type": "micro",
            "sequence": sequence,
            "description": description,
            "code": code,
            "line_start": start_line,
            "line_end": end_line,
            "injected_context": injected_context
        })
        
        sequence += 1

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        i = 0
        current_line_start = 1
        
        while i < len(lines):
            line = lines[i]
            sanitized_line = TAG_RE.sub("", line).strip()
            actual_line_number = i + 1
            
            is_header = line.strip().startswith('<c>*') and sanitized_line.startswith('*')

            if is_header:
                title_content = sanitized_line.lstrip('*').strip()
                if re.fullmatch(r"[-\s]{3,}", title_content) or len(title_content) < 3:
                    i += 1
                    continue
                
                if current_title is not None and (description_lines or code_lines):
                    finalize_micro_chunk(current_title, description_lines, code_lines, 
                                        current_line_start, actual_line_number - 1)
                    
                    # Update historical context with the code from the chunk that just finished
                    temp_defined = set(d for _, defs, _ in historical_context for d in defs)
                    for code_line in code_lines:
                        defs = get_defs_for_line(code_line, temp_defined)
                        historical_context.append((code_line, defs, actual_line_number))
                        temp_defined.update(defs)
                
                current_title = title_content
                description_lines = []
                code_lines = []
                current_line_start = actual_line_number
                
                # Look ahead for additional description lines
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    next_sanitized = TAG_RE.sub("", next_line).strip()
                    if next_line.strip().startswith('<c>*') and next_sanitized.startswith('*'):
                        desc_content = next_sanitized.lstrip('*').strip()
                        if desc_content and not desc_content.endswith(':'):
                            description_lines.append(desc_content)
                            j += 1
                        else:
                            break
                    else:
                        break
                i = j
                continue
            
            elif current_title is not None and line.strip().startswith('<l>'):
                code_lines.append(sanitized_line)
            
            elif current_title is None and line.strip().startswith('<l>'):
                # Handle global code before any titles by adding it to history
                temp_defined = set(d for _, defs, _ in historical_context for d in defs)
                defs = get_defs_for_line(sanitized_line, temp_defined)
                historical_context.append((sanitized_line, defs, actual_line_number))

            i += 1

        # Finalize last chunk
        if current_title is not None and (description_lines or code_lines):
            finalize_micro_chunk(current_title, description_lines, code_lines, 
                                current_line_start, len(lines))
            
    except (OSError, IOError) as exc:
        print(f"[ERROR] Unable to create micro chunks from {path}: {exc}")

    return micro_chunks


def process_file_for_database(path: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    Process a file and return database-ready context and chunks.
    
    Returns:
        Tuple of (file_context, all_chunks)
        - file_context: Dictionary for contexts table
        - all_chunks: List of dictionaries for chunks table (full + micro chunks)
    """
    # Generate unique context ID
    context_id = hashlib.md5(path.encode('utf-8')).hexdigest()[:8]
    
    # Extract file context
    file_context = extract_file_context(path)
    file_context["id"] = context_id  # Add the ID for database insertion
    
    # Create all chunks
    all_chunks = []
    
    # 1. Create full chunk
    full_chunk = create_full_chunk(path, context_id)
    if full_chunk:
        all_chunks.append(full_chunk)
    
    # 2. Create micro chunks
    micro_chunks = create_micro_chunks(path, context_id)
    all_chunks.extend(micro_chunks)
    
    return file_context, all_chunks


# Legacy function for backward compatibility and testing
def chunk_file(path: str, max_code_lines: int = MAX_CODE_LINES) -> List[Dict[str, str]]:
    """
    Legacy function for backward compatibility.
    Returns chunks in the old format for terminal display.
    """
    file_context, all_chunks = process_file_for_database(path)
    
    # Convert to old format for display
    display_chunks = []
    
    # Add file context as first chunk
    display_chunks.append({
        "title": f"FILE_CONTEXT: {file_context['file']}",
        "description": file_context['header'],
        "code": f"* FILE: {file_context['file']}\n* PROCEDURE: {file_context['procedure']}\n* FILE_ID: {file_context['id']}",
        "tags": f"file-context,{file_context['tags']},file-id-{file_context['id']}"
    })
    
    # Convert database chunks to display format
    for chunk in all_chunks:
        if chunk['chunk_type'] == 'full':
            title = f"FULL_CHUNK: {file_context['file']}"
        else:
            title = f"MICRO_CHUNK #{chunk['sequence']}"
            
        display_code = chunk['code']
        if chunk['injected_context']:
            display_code = chunk['injected_context'] + "\n" + chunk['code']
            
        display_chunks.append({
            "title": title,
            "description": chunk['description'],
            "code": display_code,
            "tags": f"{chunk['chunk_type']},file-id-{file_context['id']}"
        })
    
    return display_chunks


def print_chunks(path: str, chunks: List[Dict[str, str]], verbose: bool = False) -> None:
    """Print chunks to terminal with nice formatting"""
    print(f"\n{'='*60}")
    print(f"FILE: {path}")
    print(f"{'='*60}")
    
    for idx, ch in enumerate(chunks, 1):
        if not ch["title"] or not ch["code"]:
            print(f"⚠️  Missing title/code in chunk #{idx}")
            continue
            
        print(f"\n--- CHUNK #{idx}: {ch['title']} ---")
        
        if ch["description"]:
            print(f"DESCRIPTION:")
            print(ch["description"])
            print()
        
        print(f"CODE:")
        print(ch["code"])
        
        if ch["tags"]:
            print(f"TAGS: {ch['tags']}")
        
        if verbose and idx < len(chunks):
            print("-" * 40)


############################################################
# Main CLI entry-point
############################################################

def main() -> None:
    # Use configuration variables from the top of the file
    root_path = os.path.abspath(HALCON_DIR)
    patterns = [f"*.{ext.strip().lower()}" for ext in EXTENSIONS if ext.strip()]
    
    # Get list of available files
    available_files = list(scan_files(root_path, patterns))
    
    if not available_files:
        print(f"No files found in {root_path} matching patterns: {patterns}")
        return
    
    print(f"Found {len(available_files)} files matching patterns: {patterns}")
    print(f"Mode: {MODE}")
    
    if MODE == "development":
        # Development mode - process single file
        run_development_mode(available_files, root_path)
    elif MODE == "database":
        # Database mode - process all files
        run_database_mode(available_files, root_path)
    else:
        print(f"Error: Invalid MODE '{MODE}'. Must be 'development' or 'database'")


def run_development_mode(available_files: List[str], root_path: str) -> None:
    """Run in development mode - process single file for testing."""
    # Validate file number
    if FILE_NUMBER < 0 or FILE_NUMBER >= len(available_files):
        print(f"Error: FILE_NUMBER {FILE_NUMBER} is out of range.")
        print(f"Available range: 0 to {len(available_files) - 1}")
        print(f"Total files found: {len(available_files)}")
        
        # Show first 10 files as examples
        print(f"\nFirst 10 available files:")
        for i in range(min(10, len(available_files))):
            rel_path = os.path.relpath(available_files[i], root_path)
            print(f"  {i}: {rel_path}")
        if len(available_files) > 10:
            print(f"  ... and {len(available_files) - 10} more files")
        return
    
    # Process selected file
    selected_file = available_files[FILE_NUMBER]
    rel_path = os.path.relpath(selected_file, root_path)
    
    print(f"Processing file #{FILE_NUMBER}: {rel_path}")
    print(f"Full path: {selected_file}")
    print(f"Max code lines per chunk: {MAX_CODE_LINES}")
    print("Deduplication enabled - identical content will be skipped\n")
    
    if is_text_file(selected_file):
        # Clear global state before processing each file
        clear_global_state()
        
        # Show database-ready format for first few entries
        print("="*60)
        print("DATABASE-READY OUTPUT (Sample)")
        print("="*60)
        
        file_context, all_chunks = process_file_for_database(selected_file)
        
        print(f"\nCONTEXTS TABLE ENTRY:")
        print(f"ID: {file_context['id']}")
        print(f"File: {file_context['file']}")
        print(f"Procedure: {file_context['procedure']}")
        print(f"Header: {file_context['header'][:100]}..." if len(file_context['header']) > 100 else f"Header: {file_context['header']}")
        print(f"Tags: {file_context['tags']}")
        
        print(f"\nCHUNKS TABLE ENTRIES ({len(all_chunks)} total):")
        for i, chunk in enumerate(all_chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"  Context ID: {chunk['context_id']}")
            print(f"  Type: {chunk['chunk_type']}")
            print(f"  Sequence: {chunk['sequence']}")
            print(f"  Lines: {chunk['line_start']}-{chunk['line_end']}")
            print(f"  Description: {chunk['description'][:50]}..." if len(chunk['description']) > 50 else f"  Description: {chunk['description']}")
            print(f"  Code length: {len(chunk['code'])} chars")
            print(f"  Has injected context: {'Yes' if chunk['injected_context'] else 'No'}")
        
        if len(all_chunks) > 3:
            print(f"\n... and {len(all_chunks) - 3} more chunks")
        
        print("\n" + "="*60)
        print("LEGACY DISPLAY FORMAT")
        print("="*60)
        
        chunks = chunk_file(selected_file, max_code_lines=MAX_CODE_LINES)
        print_chunks(selected_file, chunks, verbose=VERBOSE)
        
        print(f"\nSUMMARY:")
        print(f"Processed 1 file: {rel_path}")
        print(f"Database entries: 1 context + {len(all_chunks)} chunks")
        print(f"  - 1 full chunk (complete file)")
        print(f"  - {len(all_chunks)-1} micro chunks (intelligent segments)")
        if DUPLICATE_COUNT > 0:
            print(f"Skipped {DUPLICATE_COUNT} duplicate chunks")
    else:
        print(f"Error: {selected_file} is not a text file")


def run_database_mode(available_files: List[str], root_path: str) -> None:
    """Run in database mode - process all files and create database."""
    # Generate unique database name
    db_path = generate_db_name()
    
    print(f"\n🚀 Starting database creation...")
    print(f"📁 Processing {len(available_files)} files")
    print(f"💾 Database: {db_path}")
    
    # Create database
    conn = create_database(db_path)
    
    total_contexts = 0
    total_chunks = 0
    total_full_chunks = 0
    total_micro_chunks = 0
    processed_files = 0
    errors = 0
    
    try:
        for i, file_path in enumerate(available_files):
            rel_path = os.path.relpath(file_path, root_path)
            
            if not is_text_file(file_path):
                print(f"⚠️  Skipping non-text file: {rel_path}")
                continue
            
            try:
                # Clear global state for each file
                clear_global_state()
                
                # Process file
                file_context, all_chunks = process_file_for_database(file_path)
                
                # Insert into database
                insert_context(conn, file_context)
                total_contexts += 1
                
                for chunk in all_chunks:
                    insert_chunk(conn, chunk)
                    total_chunks += 1
                    if chunk['chunk_type'] == 'full':
                        total_full_chunks += 1
                    else:
                        total_micro_chunks += 1
                
                processed_files += 1
                
                # Progress indicator
                if processed_files % 10 == 0 or processed_files == len(available_files):
                    print(f"📊 Progress: {processed_files}/{len(available_files)} files processed")
                
            except Exception as e:
                print(f"❌ Error processing {rel_path}: {e}")
                errors += 1
                continue
    
    finally:
        conn.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 DATABASE CREATION COMPLETE!")
    print(f"{'='*60}")
    print(f"📄 Database: {db_path}")
    print(f"📁 Files processed: {processed_files}/{len(available_files)}")
    print(f"❌ Errors: {errors}")
    print(f"📊 Total contexts: {total_contexts}")
    print(f"📦 Total chunks: {total_chunks}")
    print(f"   - Full chunks: {total_full_chunks}")
    print(f"   - Micro chunks: {total_micro_chunks}")
    if DUPLICATE_COUNT > 0:
        print(f"🔄 Duplicates skipped: {DUPLICATE_COUNT}")
    
    # Database stats
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM contexts")
        db_contexts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks")
        db_chunks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE chunk_type = 'full'")
        db_full = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE chunk_type = 'micro'")
        db_micro = cursor.fetchone()[0]
        
        print(f"\n💾 Database verification:")
        print(f"   - Contexts in DB: {db_contexts}")
        print(f"   - Chunks in DB: {db_chunks}")
        print(f"   - Full chunks in DB: {db_full}")
        print(f"   - Micro chunks in DB: {db_micro}")
        
        conn.close()
        
    except Exception as e:
        print(f"⚠️  Could not verify database: {e}")


if __name__ == "__main__":
    main() 