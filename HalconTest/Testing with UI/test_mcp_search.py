#!/usr/bin/env python3
"""
Interactive test script for HALCON MCP Server endpoints.
Allows you to select functions and input parameters dynamically.
"""

import sys
import json
from pathlib import Path

# Add the parent directory to Python path so we can import HalconTest
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all the functions from our MCP server
from HalconTest import (
    get_operators_info,
    search_operators,
    list_halcon_operators,
    search_code,
    validate_database
)

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_result(result):
    """Print result in a formatted way."""
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif isinstance(result, list):
        if result:
            print(f"Found {len(result)} results:")
            for i, item in enumerate(result, 1):
                print(f"\n--- Result {i} ---")
                if isinstance(item, dict):
                    print(json.dumps(item, indent=2, ensure_ascii=False))
                else:
                    print(item)
        else:
            print("No results found.")
    else:
        print(result)

def get_string_input(prompt, default=None):
    """Get string input with optional default."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def get_int_input(prompt, default=None, min_val=None, max_val=None):
    """Get integer input with validation."""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} [{default}]: ").strip()
                if not user_input:
                    return default
                value = int(user_input)
            else:
                value = int(input(f"{prompt}: "))
            
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
            
            return value
        except ValueError:
            print("Please enter a valid integer.")

def get_list_input(prompt, valid_options=None, default=None):
    """Get list input (comma-separated values). Handles 'all' keyword."""
    if default:
        default_str = ", ".join(default)
        user_input = input(f"{prompt} [{default_str}]: ").strip()
        if not user_input:
            return default
    else:
        user_input = input(f"{prompt}: ").strip()
    
    if not user_input:
        return []
    
    items = [item.strip().lower() for item in user_input.split(",")]

    if "all" in items:
        return ["all"]
    
    if valid_options:
        valid_items = [item for item in items if item in valid_options]
        invalid_items = [item for item in items if item not in valid_options]
        
        if invalid_items:
            print(f"Warning: Invalid options ignored: {invalid_items}")
            print(f"Valid options are: {valid_options} or 'all'")
        
        return valid_items
    
    return items

def test_get_operators_info():
    """Test the operators info resource."""
    print_header("Get Operators Info")
    try:
        result = get_operators_info()
        print("✅ Success!")
        print_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")

def test_search_operators():
    """Test unified operator search (exact and semantic)."""
    print_header("Search HALCON Operators")
    
    query = get_string_input("Enter search query (operator name or natural language)")
    if not query:
        print("Search query is required.")
        return
    
    search_mode = get_string_input("Search mode (auto/exact/semantic)", default="auto")
    
    print("\nAvailable fields: name, signature, description, parameters, results, url, or 'all'")
    fields = get_list_input(
        "Enter fields to retrieve (comma-separated)", 
        valid_options=["name", "signature", "description", "parameters", "results", "url"],
        default=["name", "signature", "description", "url"]
    )
    
    if not fields:
        print("No valid fields specified.")
        return
    
    k = get_int_input("Enter number of results (for semantic search)", default=5, min_val=1, max_val=20)
    
    try:
        result = search_operators(query=query, search_mode=search_mode, fields=fields, k=k)
        print("✅ Success!")
        print_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")

def test_list_halcon_operators():
    """Test listing operators with pagination."""
    print_header("List HALCON Operators")
    
    offset = get_int_input("Enter offset (number to skip)", default=0, min_val=0)
    limit = get_int_input("Enter limit (max results)", default=10, min_val=1, max_val=100)
    
    try:
        result = list_halcon_operators(offset=offset, limit=limit)
        print("✅ Success!")
        print_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")

def test_search_code():
    """Test unified code search (examples and chunks)."""
    print_header("Search HALCON Code")
    
    query = get_string_input("Enter search query")
    if not query:
        print("Search query is required.")
        return
    
    granularity = get_string_input("Search granularity (auto/examples/chunks/both)", default="auto")
    chunk_type = get_string_input("Chunk type when searching chunks (all/full/micro)", default="all")
    
    include_context = get_string_input("Include context? (y/n)", default="y").lower() == 'y'
    include_navigation = get_string_input("Include navigation? (y/n)", default="y").lower() == 'y'
    
    k = get_int_input("Enter number of results", default=5, min_val=1, max_val=20)
    
    try:
        results = search_code(
            query=query, 
            granularity=granularity,
            chunk_type=chunk_type,
            include_context=include_context,
            include_navigation=include_navigation,
            k=k
        )
        print("✅ Success!")
        print_result(results)
    except Exception as e:
        print(f"❌ Error: {e}")

def show_menu():
    """Display the main menu."""
    print("\n" + "🧪 HALCON MCP Server Interactive Tester")
    print("=" * 50)
    print("1. Get Operators Info (resource)")
    print("2. Search HALCON Operators (unified)")
    print("3. List HALCON Operators")
    print("4. Search HALCON Code (unified)")
    print("5. Validate Database")
    print("0. Exit")
    print("=" * 50)

def validate_db():
    """Validate database connection."""
    print_header("Database Validation")
    try:
        validate_database()
        print("✅ Database validation passed!")
    except Exception as e:
        print(f"❌ Database validation failed: {e}")

def main():
    """Main interactive loop."""
    print("🚀 Welcome to HALCON MCP Server Interactive Tester!")
    print("This tool lets you test all MCP server functions interactively.")
    
    # First validate database
    print("\nValidating database connection...")
    try:
        validate_database()
        print("✅ Database ready!")
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        print("Please ensure the database exists and is properly built.")
        return
    
    functions = {
        "1": test_get_operators_info,
        "2": test_search_operators,
        "3": test_list_halcon_operators,
        "4": test_search_code,
        "5": validate_db
    }
    
    while True:
        show_menu()
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == "0":
            print("\n👋 Goodbye! Thanks for testing the HALCON MCP Server.")
            break
        elif choice in functions:
            try:
                functions[choice]()
            except KeyboardInterrupt:
                print("\n\n⚠️  Operation cancelled by user.")
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
            
            input("\nPress Enter to continue...")
        else:
            print("❌ Invalid choice. Please enter a number between 0-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
