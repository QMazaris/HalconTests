import os
import sqlite3
import sys
import time
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Use the newer HALCON 20.11 documentation which has cleaner structure
BASE_URL = "https://www.mvtec.com/doc/halcon/2011/en/"
INDEX_RELATIVE = "index_by_name.html"
DB_PATH = Path(__file__).with_name("halcon_operators.db")


def ensure_database(db_path: Path) -> sqlite3.Connection:
    """Create the SQLite database with improved schema."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Drop existing table to recreate with new schema
    cur.execute("DROP TABLE IF EXISTS operators")
    
    cur.execute(
        """
        CREATE TABLE operators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            url TEXT,
            signature TEXT,
            description TEXT,
            parameters TEXT,
            examples TEXT,
            complexity TEXT,
            predecessors TEXT,
            successors TEXT,
            alternatives TEXT,
            module TEXT,
            full_content TEXT
        )
        """
    )
    con.commit()
    return con


def fetch_html(url: str) -> str:
    """Download HTML content, retrying on transient errors."""
    retries = 3
    backoff = 2
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2


def clean_operator_name(raw_name: str) -> str:
    """Extract clean operator name from the malformed HTML text."""
    # Remove common prefixes/suffixes and repetitions
    name = raw_name.strip()
    
    # Handle cases like "thresholdThresholdThresholdthresholdThresholdthreshold"
    # Look for pattern where name is repeated in different cases
    words = re.findall(r'[A-Z][a-z]*|[a-z]+', name)
    if words:
        # Find the most common word (likely the actual operator name)
        word_counts = {}
        for word in words:
            lower_word = word.lower()
            word_counts[lower_word] = word_counts.get(lower_word, 0) + 1
        
        if word_counts:
            # Get the most frequent word
            most_common = max(word_counts, key=word_counts.get)
            # If it appears multiple times, it's likely the operator name
            if word_counts[most_common] > 1:
                return most_common
    
    # Fallback: try to extract from beginning
    # Look for first reasonable word
    match = re.match(r'^([a-z_]+)', name.lower())
    if match:
        return match.group(1)
    
    # Last resort: return first part before any uppercase
    parts = re.split(r'[A-Z]', name)
    if parts and parts[0]:
        return parts[0].lower()
    
    return name.lower()


def parse_operator_links(index_html: str) -> List[Tuple[str, str]]:
    """Return list of (clean_name, relative_url) tuples for each operator."""
    soup = BeautifulSoup(index_html, "html.parser")
    links: List[Tuple[str, str]] = []
    
    # Look for all links that contain .html and appear to be operator references
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        raw_name = a.text.strip()
        
        # Filter for operator links
        if (href.endswith(".html") and 
            raw_name and 
            not href.startswith("index") and 
            not href.startswith("class") and
            len(raw_name) > 1 and
            not raw_name.startswith("•")):
            
            clean_name = clean_operator_name(raw_name)
            if clean_name and len(clean_name) > 1:
                links.append((clean_name, href))
    
    return links


def extract_text_content(element) -> str:
    """Extract clean text content from a BeautifulSoup element."""
    if not element:
        return ""
    return element.get_text(" ", strip=True)


def extract_comprehensive_operator_info(op_html: str) -> Dict[str, str]:
    """Extract comprehensive information from operator page HTML."""
    soup = BeautifulSoup(op_html, "html.parser")
    
    info = {
        'signature': '',
        'description': '',
        'parameters': '',
        'examples': '',
        'complexity': '',
        'predecessors': '',
        'successors': '',
        'alternatives': '',
        'module': '',
        'full_content': ''
    }
    
    # Extract signature/syntax
    signature_section = soup.find(lambda tag: tag.name in ["h2", "h3"] and tag.text and "Signature" in tag.text)
    if signature_section:
        # Get all code/pre blocks after signature header
        signatures = []
        for sibling in signature_section.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            if sibling.name in ["pre", "code"]:
                signatures.append(extract_text_content(sibling))
            elif sibling.name == "p" and sibling.find("code"):
                # Sometimes signature is in code within paragraphs
                code_elements = sibling.find_all("code")
                for code in code_elements:
                    signatures.append(extract_text_content(code))
        info['signature'] = "\n\n".join(signatures)
    
    # Extract description
    desc_section = soup.find(lambda tag: tag.name in ["h2", "h3"] and tag.text and "Description" in tag.text)
    if desc_section:
        desc_parts = []
        for sibling in desc_section.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            if sibling.name in ["p", "div"]:
                desc_parts.append(extract_text_content(sibling))
        info['description'] = "\n\n".join(desc_parts)
    
    # Extract parameters
    params_section = soup.find(lambda tag: tag.name in ["h2", "h3"] and tag.text and "Parameters" in tag.text)
    if params_section:
        param_parts = []
        for sibling in params_section.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            if sibling.name in ["p", "div", "dl", "ul", "ol"]:
                param_parts.append(extract_text_content(sibling))
        info['parameters'] = "\n\n".join(param_parts)
    
    # Extract examples
    example_sections = soup.find_all(lambda tag: tag.name in ["h2", "h3"] and tag.text and "Example" in tag.text)
    examples = []
    for example_section in example_sections:
        example_parts = []
        for sibling in example_section.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            if sibling.name in ["pre", "code"]:
                example_parts.append(extract_text_content(sibling))
            elif sibling.name == "p" and sibling.find("code"):
                code_elements = sibling.find_all("code")
                for code in code_elements:
                    example_parts.append(extract_text_content(code))
        if example_parts:
            examples.extend(example_parts)
    info['examples'] = "\n\n".join(examples)
    
    # Extract complexity
    complexity_section = soup.find(lambda tag: tag.name in ["h2", "h3"] and tag.text and "Complexity" in tag.text)
    if complexity_section:
        complexity_parts = []
        for sibling in complexity_section.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            if sibling.name in ["p", "div"]:
                complexity_parts.append(extract_text_content(sibling))
        info['complexity'] = "\n\n".join(complexity_parts)
    
    # Extract predecessors
    pred_section = soup.find(lambda tag: tag.name in ["h2", "h3"] and tag.text and "Possible Predecessors" in tag.text)
    if pred_section:
        pred_parts = []
        for sibling in pred_section.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            if sibling.name in ["p", "div"]:
                pred_parts.append(extract_text_content(sibling))
        info['predecessors'] = "\n\n".join(pred_parts)
    
    # Extract successors
    succ_section = soup.find(lambda tag: tag.name in ["h2", "h3"] and tag.text and "Possible Successors" in tag.text)
    if succ_section:
        succ_parts = []
        for sibling in succ_section.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            if sibling.name in ["p", "div"]:
                succ_parts.append(extract_text_content(sibling))
        info['successors'] = "\n\n".join(succ_parts)
    
    # Extract alternatives
    alt_section = soup.find(lambda tag: tag.name in ["h2", "h3"] and tag.text and "Alternatives" in tag.text)
    if alt_section:
        alt_parts = []
        for sibling in alt_section.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            if sibling.name in ["p", "div"]:
                alt_parts.append(extract_text_content(sibling))
        info['alternatives'] = "\n\n".join(alt_parts)
    
    # Extract module
    module_section = soup.find(lambda tag: tag.name in ["h2", "h3"] and tag.text and "Module" in tag.text)
    if module_section:
        module_parts = []
        for sibling in module_section.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            if sibling.name in ["p", "div"]:
                module_parts.append(extract_text_content(sibling))
        info['module'] = "\n\n".join(module_parts)
    
    # Store full content for fallback
    info['full_content'] = extract_text_content(soup)
    
    return info


def store_operator(con: sqlite3.Connection, name: str, url: str, info: Dict[str, str]) -> None:
    """Store operator with comprehensive information."""
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO operators 
        (name, url, signature, description, parameters, examples, complexity, 
         predecessors, successors, alternatives, module, full_content)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (name, url, info['signature'], info['description'], info['parameters'],
         info['examples'], info['complexity'], info['predecessors'], 
         info['successors'], info['alternatives'], info['module'], info['full_content']),
    )
    con.commit()


def scrape_all() -> None:
    """Scrape all HALCON operators with improved extraction."""
    print("Fetching operator index...")
    index_url = BASE_URL + INDEX_RELATIVE
    index_html = fetch_html(index_url)
    operator_links = parse_operator_links(index_html)

    print(f"Found {len(operator_links)} operators. Processing...")
    con = ensure_database(DB_PATH)

    for name, rel_url in tqdm(operator_links):
        full_url = BASE_URL + rel_url
        try:
            op_html = fetch_html(full_url)
            info = extract_comprehensive_operator_info(op_html)
            store_operator(con, name, full_url, info)
        except Exception as e:
            print(f"Failed to process {name} ({rel_url}): {e}", file=sys.stderr)

    con.close()
    print(f"Done. Improved database saved to {DB_PATH}")


if __name__ == "__main__":
    scrape_all() 