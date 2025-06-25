import sqlite3
import hashlib
from collections import defaultdict

def analyze_database():
    """Analyze the database structure and find duplicates"""
    conn = sqlite3.connect('halcon_code_examples.db')
    cursor = conn.cursor()
    
    # Get table schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables found: {tables}")
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table_name}';")
        schema = cursor.fetchone()
        print(f"\nTable '{table_name}' schema:")
        print(schema[0])
        
        # Get sample data
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"Total rows: {count}")
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        print(f"Columns: {column_names}")
        
        # Show sample rows
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        sample_rows = cursor.fetchall()
        print("Sample rows:")
        for i, row in enumerate(sample_rows):
            print(f"Row {i+1}: {row}")
    
    conn.close()

def find_duplicates():
    """Find and identify duplicate entries"""
    conn = sqlite3.connect('halcon_code_examples.db')
    cursor = conn.cursor()
    
    # Get table name (assuming there's one main table)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\nAnalyzing duplicates in table: {table_name}")
        
        # Get all rows
        cursor.execute(f"SELECT rowid, * FROM {table_name};")
        rows = cursor.fetchall()
        
        # Group by content hash to find duplicates
        content_hash_groups = defaultdict(list)
        
        for row in rows:
            rowid = row[0]
            content = str(row[1:])  # All columns except rowid
            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_hash_groups[content_hash].append((rowid, row))
        
        # Find duplicates
        duplicates = {k: v for k, v in content_hash_groups.items() if len(v) > 1}
        
        print(f"Found {len(duplicates)} groups of duplicates")
        print(f"Total duplicate rows: {sum(len(group) - 1 for group in duplicates.values())}")
        
        # Show some examples
        for i, (hash_key, group) in enumerate(duplicates.items()):
            if i >= 3:  # Show only first 3 examples
                break
            print(f"\nDuplicate group {i+1} (hash: {hash_key[:8]}...):")
            print(f"  {len(group)} identical rows")
            print(f"  Row IDs: {[row[0] for row in group]}")
            if len(group[0][1]) > 1:  # Show content if not too long
                content_preview = str(group[0][1][1])[:100] + "..." if len(str(group[0][1][1])) > 100 else str(group[0][1][1])
                print(f"  Content preview: {content_preview}")
    
    conn.close()
    return duplicates, table_name

def remove_duplicates():
    """Remove duplicate entries, keeping only one instance of each"""
    conn = sqlite3.connect('halcon_code_examples.db')
    cursor = conn.cursor()
    
    # Get table name
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    total_removed = 0
    
    for table in tables:
        table_name = table[0]
        print(f"\nRemoving duplicates from table: {table_name}")
        
        # Get all rows
        cursor.execute(f"SELECT rowid, * FROM {table_name};")
        rows = cursor.fetchall()
        
        # Group by content hash
        content_hash_groups = defaultdict(list)
        
        for row in rows:
            rowid = row[0]
            content = str(row[1:])  # All columns except rowid
            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_hash_groups[content_hash].append(rowid)
        
        # Remove duplicates (keep first occurrence)
        rows_to_delete = []
        for group in content_hash_groups.values():
            if len(group) > 1:
                # Keep the first one, delete the rest
                rows_to_delete.extend(group[1:])
        
        print(f"Removing {len(rows_to_delete)} duplicate rows...")
        
        # Delete duplicates
        for rowid in rows_to_delete:
            cursor.execute(f"DELETE FROM {table_name} WHERE rowid = ?", (rowid,))
        
        total_removed += len(rows_to_delete)
        
        # Get new count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        new_count = cursor.fetchone()[0]
        print(f"Rows remaining: {new_count}")
    
    # Commit changes and vacuum to reclaim space
    conn.commit()
    print(f"\nVacuuming database to reclaim space...")
    cursor.execute("VACUUM;")
    conn.commit()
    conn.close()
    
    print(f"\nTotal duplicate rows removed: {total_removed}")
    return total_removed

if __name__ == "__main__":
    print("=== Analyzing Database ===")
    analyze_database()
    
    print("\n=== Finding Duplicates ===")
    duplicates, table_name = find_duplicates()
    
    if duplicates:
        response = input(f"\nFound duplicates. Do you want to remove them? (y/n): ")
        if response.lower() == 'y':
            print("\n=== Removing Duplicates ===")
            removed_count = remove_duplicates()
            print(f"\nDatabase cleanup complete! Removed {removed_count} duplicate entries.")
        else:
            print("Duplicate removal cancelled.")
    else:
        print("\nNo duplicates found!") 