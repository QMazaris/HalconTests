import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.mvtec.com/doc/halcon/11/en/"
INDEX_RELATIVE = "index_by_name.html"
DB_PATH = Path(__file__).with_name("halcon_operators.db")


def ensure_database(db_path: Path) -> sqlite3.Connection:
    """Create the SQLite database and required table if they do not exist."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS operators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            url TEXT,
            syntax TEXT,
            description TEXT
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


def parse_operator_links(index_html: str) -> List[Tuple[str, str]]:
    """Return list of (name, relative_url) tuples for each operator."""
    soup = BeautifulSoup(index_html, "html.parser")
    links: List[Tuple[str, str]] = []
    # Look for all links that contain .html and appear to be operator references
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        name = a.text.strip()
        # Filter for operator links - they typically have alphanumeric names and end with .html
        if (href.endswith(".html") and 
            name and 
            not href.startswith("index") and 
            not href.startswith("class") and
            len(name) > 1 and
            not name.startswith("•")):
            # Clean up the name
            name = name.replace("\n", " ").strip()
            if name:
                links.append((name, href))
    return links


def extract_operator_info(op_html: str) -> Tuple[str, str]:
    """Extract syntax and description from operator page HTML."""
    soup = BeautifulSoup(op_html, "html.parser")

    # Syntax is often within <pre> following an h2/h3 with text 'Syntax'.
    syntax = ""
    syntax_header = soup.find(lambda tag: tag.name in ["h2", "h3"] and "Syntax" in tag.text)
    if syntax_header:
        pre = syntax_header.find_next("pre")
        if pre:
            syntax = pre.get_text("\n", strip=True)

    # Description: first <p> after the main heading
    description = ""
    main_heading = soup.find(["h1", "h2"], string=True)
    if main_heading:
        p = main_heading.find_next("p")
        if p:
            description = p.get_text(" ", strip=True)

    return syntax, description


def store_operator(con: sqlite3.Connection, name: str, url: str, syntax: str, description: str) -> None:
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO operators (name, url, syntax, description)
        VALUES (?, ?, ?, ?)
        """,
        (name, url, syntax, description),
    )
    con.commit()


def scrape_all() -> None:
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
            syntax, description = extract_operator_info(op_html)
            store_operator(con, name, full_url, syntax, description)
        except Exception as e:
            print(f"Failed to process {name} ({rel_url}): {e}", file=sys.stderr)

    con.close()
    print(f"Done. Database saved to {DB_PATH}")


if __name__ == "__main__":
    scrape_all() 