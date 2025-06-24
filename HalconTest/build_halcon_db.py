#!/usr/bin/env python3
"""
build_halcon_db.py

Combines operator discovery and data extraction to build a complete
HALCON operators database with name, signature, description, and page_dump.

Usage:
    python build_halcon_db.py [-v]
"""
import argparse
import csv
import sqlite3
import sys
from pathlib import Path

# Import functions from our existing modules
from scrapy import discover, ROOTS
from dataExtractor import parse_operator_page


def create_database(db_path: Path) -> None:
    """Create the database with the required schema."""
    print(f"Creating database at {db_path}")
    
    # Remove existing database if it exists
    if db_path.exists():
        db_path.unlink()
    
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE operators (
                name TEXT PRIMARY KEY,
                signature TEXT,
                description TEXT,
                page_dump TEXT,
                url TEXT
            )
        """)
        con.commit()
        print("Database schema created successfully")
    finally:
        con.close()


def discover_all_operators(verbose: bool = False) -> dict[str, str]:
    """Discover all operators from all HALCON versions."""
    print("Discovering HALCON operators...")
    
    all_ops = {}
    for key, url in ROOTS.items():
        print(f"Discovering operators from {key}...")
        ops = discover(key, url, verbose)
        
        # Merge ops, preferring newer versions for duplicates
        for name, op_url in ops.items():
            # Prefer newer versions for duplicates: dl1905 > h1811 > h12
            if name not in all_ops or key in {"dl1905", "h1811"}:
                all_ops[name] = op_url
    
    print(f"Total unique operators discovered: {len(all_ops)}")
    return all_ops


def extract_and_store_operators(operators: dict[str, str], db_path: Path, verbose: bool = False) -> None:
    """Extract data for each operator and store in the database."""
    print("Extracting operator data and building database...")
    
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        
        total = len(operators)
        success_count = 0
        error_count = 0
        
        for i, (name, url) in enumerate(operators.items(), 1):
            try:
                if verbose:
                    print(f"[{i}/{total}] Processing {name}...")
                else:
                    # Progress indicator
                    if i % 10 == 0 or i == total:
                        print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
                
                # Extract operator data
                data = parse_operator_page(url)
                
                # Store in database
                cur.execute("""
                    INSERT OR REPLACE INTO operators 
                    (name, signature, description, page_dump, url)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    data["name"],
                    data["signature"],
                    data["description"],
                    data["page_dump"],
                    url
                ))
                
                success_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"ERROR processing {name}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        con.commit()
        print(f"\nDatabase build complete!")
        print(f"Success: {success_count} operators")
        print(f"Errors: {error_count} operators")
        
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description="Build HALCON operators database")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("-o", "--output", default="halcon_operators.db", help="Output SQLite DB filename")
    args = parser.parse_args()
    
    db_path = Path(args.output)
    
    try:
        # Step 1: Create database
        create_database(db_path)
        
        # Step 2: Discover all operators
        operators = discover_all_operators(args.verbose)
        
        if not operators:
            print("No operators discovered. Exiting.")
            sys.exit(1)
        
        # Step 3: Extract data and populate database
        extract_and_store_operators(operators, db_path, args.verbose)
        
        # Step 4: Verify the database
        con = sqlite3.connect(db_path)
        try:
            cur = con.cursor()
            count = cur.execute("SELECT COUNT(*) FROM operators").fetchone()[0]
            print(f"\nFinal database contains {count} operators")
            
            # Show sample of what was stored
            sample = cur.execute("""
                SELECT name, 
                       LENGTH(signature) as sig_len,
                       LENGTH(description) as desc_len, 
                       LENGTH(page_dump) as dump_len
                FROM operators 
                ORDER BY name 
                LIMIT 5
            """).fetchall()
            
            print("\nSample entries:")
            for row in sample:
                print(f"  {row[0]}: sig={row[1]} chars, desc={row[2]} chars, dump={row[3]} chars")
                
        finally:
            con.close()
            
    except KeyboardInterrupt:
        print("\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 