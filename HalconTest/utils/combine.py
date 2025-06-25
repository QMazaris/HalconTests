import sqlite3
import shutil
import os
import time

# === CONFIGURE THESE PATHS ===
V3_DB      = "../halcon_operators_v3.db"
LOCAL_DB   = "halcon_operators_local.db"
COMBINED_DB= "combined.db"
# =============================

def merge_dbs(v3_path, local_path, out_path):
    # 1) Remove existing output file if it exists
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except PermissionError:
            print(f"Warning: Could not remove {out_path}, it may be in use. Trying alternative name...")
            out_path = f"combined_{int(time.time())}.db"
            print(f"Using output file: {out_path}")
    
    # 2) Create new combined database
    con = sqlite3.connect(out_path)
    cur = con.cursor()
    
    # 3) Create unified operators table with all necessary columns
    print("Creating unified operators table...")
    cur.execute('''
        CREATE TABLE operators (
            name TEXT PRIMARY KEY,
            signature TEXT,       
            description TEXT,     
            parameters TEXT,      
            results TEXT,
            url TEXT
        )
    ''')
    
    # 4) Attach both source databases
    cur.execute("ATTACH DATABASE ? AS v3;", (v3_path,))
    cur.execute("ATTACH DATABASE ? AS loc;", (local_path,))
    
    # 5) Insert operators from V3 database
    print("Inserting operators from V3 database...")
    cur.execute('''
        INSERT INTO operators (name, signature, description, parameters, results, url)
        SELECT name, signature, description, parameters, results, url
        FROM v3.operators
    ''')
    
    operators_count = cur.rowcount
    print(f"  Added {operators_count} operators from V3")
    
    # 6) Insert procedures from V3 database (only has id, name, signature)
    print("Inserting procedures from V3 database...")
    cur.execute('''
        INSERT OR IGNORE INTO operators (name, signature, description, parameters, results, url)
        SELECT name, signature, NULL, NULL, NULL, NULL
        FROM v3.procedures
    ''')
    
    v3_procedures_added = cur.rowcount
    print(f"  Added {v3_procedures_added} procedures from V3 (limited columns)")
    
    # 7) Update procedures from Local database (has full columns: description, parameters, results, url)
    # This will overwrite the V3 procedure data with the richer local data
    print("Updating/inserting procedures from Local database...")
    cur.execute('''
        INSERT OR REPLACE INTO operators (name, signature, description, parameters, results, url)
        SELECT name, signature, description, parameters, results, url
        FROM loc.procedures
    ''')
    
    local_procedures_added = cur.rowcount
    print(f"  Updated/added {local_procedures_added} procedures from Local (full data)")
    
    # 8) Get final count
    cur.execute('SELECT COUNT(*) FROM operators')
    total_count = cur.fetchone()[0]
    
    # 9) Check how many have full data vs limited data
    cur.execute('SELECT COUNT(*) FROM operators WHERE description IS NOT NULL')
    with_description = cur.fetchone()[0]
    
    cur.execute('SELECT COUNT(*) FROM operators WHERE description IS NULL')
    without_description = cur.fetchone()[0]
    
    # 10) Print final summary
    print("\n=== MERGE SUMMARY ===")
    print(f"V3 Operators added: {operators_count}")
    print(f"V3 Procedures added: {v3_procedures_added}")
    print(f"Local Procedures updated/added: {local_procedures_added}")
    print(f"Total HALCON functions: {total_count}")
    print(f"Functions with full data (description, etc.): {with_description}")
    print(f"Functions with limited data: {without_description}")
    
    # Verify we got the expected count
    expected = 2282 + 432  # operators + unique procedures
    if total_count == expected:
        print(f"✅ Perfect! Got expected count of {expected} functions")
    else:
        print(f"⚠️  Got {total_count} functions, expected {expected}")
        print(f"   Difference likely due to duplicates being removed")
    
    # 11) Cleanup
    con.commit()
    cur.execute("DETACH DATABASE v3;")
    cur.execute("DETACH DATABASE loc;")
    con.close()
    print(f"✅ Done — All HALCON functions merged into {out_path}")

if __name__ == "__main__":
    merge_dbs(V3_DB, LOCAL_DB, COMBINED_DB)
