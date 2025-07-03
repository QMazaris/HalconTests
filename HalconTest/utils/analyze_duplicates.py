#!/usr/bin/env python3
"""
Analyze the HALCON chunks database for duplicate content and optimization opportunities.
"""

import sqlite3
import hashlib
from collections import Counter
import sys
import os

def analyze_database(db_path: str):
    """Analyze the database for duplicate content."""
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"🔍 Analyzing database: {db_path}")
        print("=" * 60)
        
        # 1. Check for duplicate code chunks
        print("\n1. 📝 DUPLICATE CODE ANALYSIS:")
        cursor.execute('''
            SELECT code, COUNT(*) as count 
            FROM chunks 
            GROUP BY code 
            HAVING count > 1 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        duplicate_codes = cursor.fetchall()
        
        if duplicate_codes:
            total_duplicate_code_chunks = sum(count for _, count in duplicate_codes)
            print(f"   Found {len(duplicate_codes)} unique duplicate code patterns")
            print(f"   Total duplicate code chunks: {total_duplicate_code_chunks}")
            print("   Top duplicates:")
            for i, (code, count) in enumerate(duplicate_codes[:5], 1):
                code_preview = code.replace('\n', ' ')[:60] + "..." if len(code) > 60 else code
                print(f"   {i}. {count} instances: \"{code_preview}\"")
        else:
            print("   ✅ No duplicate code chunks found")

        # 2. Check for duplicate descriptions
        print("\n2. 📋 DUPLICATE DESCRIPTIONS:")
        cursor.execute('''
            SELECT description, COUNT(*) as count 
            FROM chunks 
            WHERE description != "" 
            GROUP BY description 
            HAVING count > 1 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        duplicate_descriptions = cursor.fetchall()
        
        if duplicate_descriptions:
            total_duplicate_desc_chunks = sum(count for _, count in duplicate_descriptions)
            print(f"   Found {len(duplicate_descriptions)} unique duplicate descriptions")
            print(f"   Total duplicate description chunks: {total_duplicate_desc_chunks}")
            print("   Top duplicates:")
            for i, (desc, count) in enumerate(duplicate_descriptions[:5], 1):
                desc_preview = desc.replace('\n', ' ')[:60] + "..." if len(desc) > 60 else desc
                print(f"   {i}. {count} instances: \"{desc_preview}\"")
        else:
            print("   ✅ No duplicate descriptions found")

        # 3. Check for duplicate injected contexts
        print("\n3. 🔗 DUPLICATE INJECTED CONTEXTS:")
        cursor.execute('''
            SELECT injected_context, COUNT(*) as count 
            FROM chunks 
            WHERE injected_context != "" 
            GROUP BY injected_context 
            HAVING count > 1 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        duplicate_contexts = cursor.fetchall()
        
        if duplicate_contexts:
            total_duplicate_context_chunks = sum(count for _, count in duplicate_contexts)
            print(f"   Found {len(duplicate_contexts)} unique duplicate contexts")
            print(f"   Total duplicate context chunks: {total_duplicate_context_chunks}")
            print("   Top duplicates:")
            for i, (context, count) in enumerate(duplicate_contexts[:3], 1):
                context_preview = context.replace('\n', ' ')[:60] + "..." if len(context) > 60 else context
                print(f"   {i}. {count} instances: \"{context_preview}\"")
        else:
            print("   ✅ No duplicate injected contexts found")

        # 4. Check for similar file headers
        print("\n4. 📄 DUPLICATE HEADERS:")
        cursor.execute('''
            SELECT header, COUNT(*) as count 
            FROM contexts 
            WHERE header != "" 
            GROUP BY header 
            HAVING count > 1 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        duplicate_headers = cursor.fetchall()
        
        if duplicate_headers:
            total_duplicate_headers = sum(count for _, count in duplicate_headers)
            print(f"   Found {len(duplicate_headers)} unique duplicate headers")
            print(f"   Total duplicate header files: {total_duplicate_headers}")
            print("   Top duplicates:")
            for i, (header, count) in enumerate(duplicate_headers[:3], 1):
                header_preview = header.replace('\n', ' ')[:60] + "..." if len(header) > 60 else header
                print(f"   {i}. {count} instances: \"{header_preview}\"")
        else:
            print("   ✅ No duplicate headers found")

        # 5. Check for common code patterns
        print("\n5. 🔍 COMMON CODE PATTERNS:")
        cursor.execute('''
            SELECT SUBSTR(code, 1, 50) as code_start, COUNT(*) as count 
            FROM chunks 
            WHERE LENGTH(code) > 20
            GROUP BY code_start 
            HAVING count > 5 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        common_patterns = cursor.fetchall()
        
        if common_patterns:
            print(f"   Found {len(common_patterns)} common code patterns (5+ occurrences):")
            for i, (pattern, count) in enumerate(common_patterns[:5], 1):
                print(f"   {i}. {count} instances start with: \"{pattern}...\"")
        else:
            print("   ✅ No significant common patterns found")

        # 6. Overall statistics
        print("\n" + "=" * 60)
        print("📊 OVERALL STATISTICS:")
        
        cursor.execute('SELECT COUNT(*) FROM chunks')
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM contexts')
        total_contexts = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT code) FROM chunks')
        unique_codes = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT description) FROM chunks WHERE description != ""')
        unique_descriptions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM chunks WHERE description != ""')
        total_descriptions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM chunks WHERE injected_context != ""')
        total_injected = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT injected_context) FROM chunks WHERE injected_context != ""')
        unique_injected = cursor.fetchone()[0]
        
        print(f"   Total contexts: {total_contexts}")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Unique code patterns: {unique_codes}")
        print(f"   Code uniqueness: {unique_codes/total_chunks*100:.1f}%")
        
        if total_descriptions > 0:
            print(f"   Total descriptions: {total_descriptions}")
            print(f"   Unique descriptions: {unique_descriptions}")
            print(f"   Description uniqueness: {unique_descriptions/total_descriptions*100:.1f}%")
        
        if total_injected > 0:
            print(f"   Total injected contexts: {total_injected}")
            print(f"   Unique injected contexts: {unique_injected}")
            print(f"   Context uniqueness: {unique_injected/total_injected*100:.1f}%")

        # 7. Optimization recommendations
        print("\n" + "=" * 60)
        print("💡 OPTIMIZATION RECOMMENDATIONS:")
        
        potential_savings = 0
        recommendations = []
        
        if duplicate_codes:
            code_savings = sum(count - 1 for _, count in duplicate_codes)
            potential_savings += code_savings
            recommendations.append(f"   🔄 Deduplicate code chunks: Could save {code_savings} records")
        
        if duplicate_descriptions:
            desc_savings = sum(count - 1 for _, count in duplicate_descriptions)
            recommendations.append(f"   📝 Normalize descriptions: {desc_savings} redundant descriptions")
        
        if duplicate_contexts:
            context_savings = sum(count - 1 for _, count in duplicate_contexts)
            recommendations.append(f"   🔗 Optimize injected contexts: {context_savings} redundant contexts")
        
        if recommendations:
            for rec in recommendations:
                print(rec)
            print(f"\n   💾 Potential space savings: {potential_savings} records ({potential_savings/total_chunks*100:.1f}% reduction)")
        else:
            print("   ✅ Database is already well-optimized!")
            print("   ✅ No significant redundancies found")

        conn.close()
        
    except Exception as e:
        print(f"❌ Error analyzing database: {e}")

def main():
    # Find the most recent database file
    db_files = [f for f in os.listdir('.') if f.startswith('halcon_chunks_') and f.endswith('.db')]
    if not db_files:
        print("❌ No halcon_chunks database files found in current directory")
        return
    
    # Use the most recent one
    latest_db = sorted(db_files)[-1]
    analyze_database(latest_db)

if __name__ == "__main__":
    main() 