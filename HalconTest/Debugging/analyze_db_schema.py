#!/usr/bin/env python3
"""Temporary script to analyze database schemas and content."""

import sqlite3
from pathlib import Path

def analyze_database(db_path, name):
    """Analyze a database and print its schema and basic stats."""
    print(f"\n{'='*50}")
    print(f"Analyzing {name}: {db_path}")
    print(f"{'='*50}")
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Get all tables
        tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"Tables found: {[t[0] for t in tables]}")
        
        for table_name, in tables:
            print(f"\n--- Table: {table_name} ---")
            
            # Get schema
            schema = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
            print("Columns:")
            for col in schema:
                print(f"  {col[1]} ({col[2]})")
            
            # Get count
            count = cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"Row count: {count}")
            
            # Sample a few rows
            if count > 0:
                sample = cur.execute(f"SELECT * FROM {table_name} LIMIT 2").fetchall()
                print("Sample rows:")
                for i, row in enumerate(sample):
                    print(f"  Row {i+1}: {dict(zip([col[1] for col in schema], row))}")
        
        con.close()
        
    except Exception as e:
        print(f"❌ Error analyzing database: {e}")

# Analyze all databases
analyze_database("databases/halcon_chunks_latest.db", "Chunks Latest")
analyze_database("combined.db", "Combined")
analyze_database("halcon_code_examplesV2.db", "Code Examples V2") 