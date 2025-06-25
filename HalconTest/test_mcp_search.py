#!/usr/bin/env python3
"""
Test script for HALCON MCP Server endpoints.
Tests all available tools and resources by calling them directly.
"""

import sys
from pathlib import Path

# Add the current directory to Python path so we can import HalconTest
sys.path.insert(0, str(Path(__file__).parent))

# Import all the functions from our MCP server
from HalconTest import (
    get_operators_info,
    search_halcon_operators, 
    get_halcon_operator,
    list_halcon_operators,
    semantic_match,
    semantic_code_search,
    validate_database
)

def test_resource():
    """Test the MCP resource endpoint."""
    print("=" * 60)
    print("TESTING RESOURCE: halcon://operators")
    print("=" * 60)
    
    try:
        info = get_operators_info()
        print("✅ Success!")
        print(f"Result: {info}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

def test_search_operators():
    """Test fuzzy search functionality."""
    print("=" * 60)
    print("TESTING TOOL: search_halcon_operators")
    print("=" * 60)
    
    test_queries = [
        ("edge detection", 3),
        ("threshold", 5),
        ("morphology", 3),
        ("read image", 2)
    ]
    
    for query, limit in test_queries:
        print(f"Query: '{query}' (limit={limit})")
        try:
            result = search_halcon_operators(query=query, limit=limit)
            print("✅ Success!")
            print(result[:300] + "..." if len(result) > 300 else result)
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)

def test_get_operator():
    """Test getting specific operator information."""
    print("=" * 60)
    print("TESTING TOOL: get_halcon_operator")
    print("=" * 60)
    
    test_cases = [
        ("threshold", "signature"),
        ("read_image", "info"), 
        ("edges_image", "info"),
        ("median_image", "full")
    ]
    
    for name, detail in test_cases:
        print(f"Operator: '{name}' (detail='{detail}')")
        try:
            result = get_halcon_operator(name=name, detail=detail)
            print("✅ Success!")
            print(result[:400] + "..." if len(result) > 400 else result)
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)

def test_list_operators():
    """Test listing operators with pagination."""
    print("=" * 60)
    print("TESTING TOOL: list_halcon_operators")
    print("=" * 60)
    
    test_cases = [
        (0, 5),    # First 5
        (10, 3),   # Skip 10, get 3
        (100, 5)   # Skip 100, get 5
    ]
    
    for offset, limit in test_cases:
        print(f"List operators (offset={offset}, limit={limit})")
        try:
            result = list_halcon_operators(offset=offset, limit=limit)
            print("✅ Success!")
            print(result[:400] + "..." if len(result) > 400 else result)
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)

def test_semantic_match():
    """Test semantic operator matching."""
    print("=" * 60)
    print("TESTING TOOL: semantic_match")
    print("=" * 60)
    
    test_queries = [
        ("how to find edges in an image", 3, "info"),
        ("image filtering and smoothing", 2, "signature"),
        ("morphological operations", 3, "info"),
        ("read and write images", 2, "full")
    ]
    
    for query, k, detail in test_queries:
        print(f"Query: '{query}' (k={k}, detail='{detail}')")
        try:
            results = semantic_match(query=query, k=k, detail=detail)
            print("✅ Success!")
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get('name', 'Unknown')} (score: {result.get('score', 0):.4f})")
                if 'description' in result:
                    desc = result['description'][:100] + "..." if len(result['description']) > 100 else result['description']
                    print(f"     Description: {desc}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)

def test_semantic_code_search():
    """Test semantic code example search."""
    print("=" * 60)
    print("TESTING TOOL: semantic_code_search")
    print("=" * 60)
    
    test_queries = [
        ("how to find edges in an image", 2),
        ("image processing pipeline", 3),
        ("morphological operations", 2),
        ("template matching", 2)
    ]
    
    for query, k in test_queries:
        print(f"Query: '{query}' (k={k})")
        try:
            results = semantic_code_search(query=query, k=k)
            print("✅ Success!")
            print(f"Found {len(results)} code examples:")
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Untitled')
                score = result.get('score', 0)
                code_preview = result.get('code', '')[:100].replace('\n', ' ') + "..."
                print(f"  {i}. {title} (score: {score:.4f})")
                print(f"     Code: {code_preview}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)

def main():
    """Run all tests."""
    print("🧪 HALCON MCP Server Test Suite")
    print("Testing all available tools and resources...\n")
    
    # First validate the database
    print("Validating database...")
    try:
        validate_database()
        print("✅ Database validation passed!\n")
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        print("Please ensure the database exists and is properly built.\n")
        return
    
    # Test all endpoints
    test_resource()
    test_search_operators()
    test_get_operator()
    test_list_operators()
    test_semantic_match()
    test_semantic_code_search()
    
    print("🎉 Test suite completed!")
    print("\nAll functions tested. Check the output above for any errors.")
    print("This demonstrates how AI models would interact with your MCP server.")

if __name__ == "__main__":
    main()
