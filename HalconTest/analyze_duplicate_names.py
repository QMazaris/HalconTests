import sqlite3
import hashlib
from collections import defaultdict, Counter
import difflib

def analyze_duplicate_titles():
    """Analyze duplicate titles and their variations"""
    conn = sqlite3.connect('halcon_code_examples.db')
    cursor = conn.cursor()
    
    # Get all titles
    cursor.execute("SELECT id, title, description, code, tags FROM examples ORDER BY title;")
    rows = cursor.fetchall()
    
    print(f"Total entries: {len(rows)}")
    
    # Group by exact title matches
    title_groups = defaultdict(list)
    for row in rows:
        id_val, title, description, code, tags = row
        title_groups[title].append(row)
    
    # Find exact title duplicates
    exact_duplicates = {title: entries for title, entries in title_groups.items() if len(entries) > 1}
    
    print(f"\nExact title duplicates: {len(exact_duplicates)} unique titles")
    total_duplicate_entries = sum(len(entries) - 1 for entries in exact_duplicates.values())
    print(f"Total duplicate entries that could be removed: {total_duplicate_entries}")
    
    # Show some examples of exact duplicates
    print("\n=== Examples of Exact Title Duplicates ===")
    for i, (title, entries) in enumerate(list(exact_duplicates.items())[:5]):
        print(f"\nTitle: '{title}' ({len(entries)} entries)")
        for j, entry in enumerate(entries[:3]):  # Show first 3
            id_val, _, description, code, tags = entry
            print(f"  Entry {j+1} (ID: {id_val}):")
            print(f"    Description: {description[:100]}{'...' if len(description) > 100 else ''}")
            print(f"    Code: {code[:100]}{'...' if len(code) > 100 else ''}")
            print(f"    Tags: {tags}")
    
    # Analyze content similarity for entries with same title
    print("\n=== Content Analysis for Duplicate Titles ===")
    identical_content = 0
    similar_content = 0
    different_content = 0
    
    for title, entries in exact_duplicates.items():
        if len(entries) <= 1:
            continue
            
        # Compare content (description + code)
        contents = []
        for entry in entries:
            content = f"{entry[2]}\n{entry[3]}"  # description + code
            contents.append(content)
        
        # Check if all contents are identical
        if all(content == contents[0] for content in contents):
            identical_content += len(entries) - 1
        else:
            # Check similarity
            is_similar = True
            for i in range(1, len(contents)):
                similarity = difflib.SequenceMatcher(None, contents[0], contents[i]).ratio()
                if similarity < 0.8:  # Less than 80% similar
                    is_similar = False
                    break
            
            if is_similar:
                similar_content += len(entries) - 1
            else:
                different_content += len(entries) - 1
    
    print(f"Entries with identical content: {identical_content}")
    print(f"Entries with similar content (>80%): {similar_content}")
    print(f"Entries with different content: {different_content}")
    
    # Find near-duplicate titles (fuzzy matching)
    print("\n=== Near-Duplicate Title Analysis ===")
    unique_titles = list(title_groups.keys())
    near_duplicates = defaultdict(list)
    
    for i, title1 in enumerate(unique_titles):
        if i % 1000 == 0:
            print(f"Processing title {i}/{len(unique_titles)}...")
        
        for j, title2 in enumerate(unique_titles[i+1:], i+1):
            similarity = difflib.SequenceMatcher(None, title1.lower(), title2.lower()).ratio()
            if similarity > 0.85:  # Very similar titles
                near_duplicates[title1].append((title2, similarity))
    
    print(f"Found {len(near_duplicates)} groups of near-duplicate titles")
    
    # Show examples
    print("\n=== Examples of Near-Duplicate Titles ===")
    for i, (base_title, similar_titles) in enumerate(list(near_duplicates.items())[:3]):
        print(f"\nBase title: '{base_title}'")
        for similar_title, similarity in similar_titles[:3]:
            print(f"  Similar: '{similar_title}' (similarity: {similarity:.3f})")
    
    conn.close()
    return exact_duplicates, near_duplicates, identical_content, similar_content, different_content

def assess_inference_impact():
    """Assess the impact of removing duplicates on inference quality"""
    print("\n=== INFERENCE IMPACT ASSESSMENT ===")
    
    print("""
REMOVING DUPLICATE TITLES - PROS AND CONS:

PROS:
✓ Reduced database size -> faster vector search
✓ Less redundant information -> cleaner embeddings
✓ Faster inference due to smaller search space
✓ Better diversity in search results
✓ Reduced memory usage during vector DB building

CONS:
⚠ Might lose valuable context variations
⚠ Could reduce recall if different implementations exist
⚠ Risk of losing edge cases or alternative approaches

RECOMMENDATION:
""")

def safe_duplicate_removal():
    """Safely remove duplicates while preserving valuable variations"""
    conn = sqlite3.connect('halcon_code_examples.db')
    cursor = conn.cursor()
    
    # Get exact title duplicates with identical content
    cursor.execute("""
        SELECT title, COUNT(*) as count
        FROM examples 
        GROUP BY title 
        HAVING COUNT(*) > 1
        ORDER BY count DESC
    """)
    
    duplicate_titles = cursor.fetchall()
    
    safe_to_remove = []
    
    for title, count in duplicate_titles:
        cursor.execute("SELECT id, description, code, tags FROM examples WHERE title = ?", (title,))
        entries = cursor.fetchall()
        
        # Group by content hash
        content_groups = defaultdict(list)
        for entry in entries:
            content = f"{entry[1]}\n{entry[2]}\n{entry[3]}"  # description + code + tags
            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_groups[content_hash].append(entry[0])  # Store ID
        
        # Mark duplicates for safe removal (keep one of each unique content)
        for content_hash, ids in content_groups.items():
            if len(ids) > 1:
                safe_to_remove.extend(ids[1:])  # Keep first, remove rest
    
    print(f"\nSafe to remove: {len(safe_to_remove)} entries")
    print(f"This would reduce the database from 61,021 to {61021 - len(safe_to_remove)} entries")
    
    conn.close()
    return safe_to_remove

def remove_safe_duplicates(ids_to_remove):
    """Remove the safely identified duplicates"""
    if not ids_to_remove:
        print("No safe duplicates to remove.")
        return
    
    conn = sqlite3.connect('halcon_code_examples.db')
    cursor = conn.cursor()
    
    print(f"Removing {len(ids_to_remove)} duplicate entries...")
    
    # Remove in batches for better performance
    batch_size = 1000
    for i in range(0, len(ids_to_remove), batch_size):
        batch = ids_to_remove[i:i + batch_size]
        placeholders = ','.join(['?' for _ in batch])
        cursor.execute(f"DELETE FROM examples WHERE id IN ({placeholders})", batch)
        print(f"Removed batch {i//batch_size + 1}/{(len(ids_to_remove) + batch_size - 1)//batch_size}")
    
    conn.commit()
    
    # Vacuum to reclaim space
    print("Vacuuming database...")
    cursor.execute("VACUUM")
    conn.commit()
    
    # Get final count
    cursor.execute("SELECT COUNT(*) FROM examples")
    final_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"Database cleanup complete!")
    print(f"Final entry count: {final_count}")
    
    return final_count

if __name__ == "__main__":
    print("=== ANALYZING DUPLICATE TITLES ===")
    exact_duplicates, near_duplicates, identical, similar, different = analyze_duplicate_titles()
    
    assess_inference_impact()
    
    print(f"""
SAFE REMOVAL STRATEGY:
- Remove entries with identical titles AND identical content: {identical} entries
- Keep entries with same title but different content (valuable variations)
- This preserves semantic diversity while reducing redundancy
""")
    
    print("\n=== IDENTIFYING SAFE DUPLICATES ===")
    safe_ids = safe_duplicate_removal()
    
    if safe_ids:
        response = input(f"\nDo you want to remove {len(safe_ids)} safe duplicate entries? (y/n): ")
        if response.lower() == 'y':
            remove_safe_duplicates(safe_ids)
        else:
            print("Duplicate removal cancelled.")
    else:
        print("No safe duplicates found for removal.") 