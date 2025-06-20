import json
from pathlib import Path
import sys

# Ensure we can import HalconTest.py easily
sys.path.append(str(Path(__file__).parent))

import HalconTest as ht

print("=== get_operators_info ===")
print(ht.get_operators_info())
print()

print("=== list_halcon_operators (first 5) ===")
listing = ht.list_halcon_operators(limit=5)
print(listing)
print()

# Extract first operator name from listing
first_name = None
for line in listing.splitlines():
    if line.startswith("**") and line.endswith("**"):
        first_name = line.strip("*\n")
        break

if first_name:
    print("=== get_halcon_operator for", first_name, "===")
    print(ht.get_halcon_operator(first_name))
    print()

    print("=== get_halcon_syntax for", first_name, "===")
    print(ht.get_halcon_syntax(first_name))
    print()
else:
    print("Could not extract operator name from listing")

print("=== search_halcon_operators for 'threshold' (limit 3) ===")
print(ht.search_halcon_operators("threshold", limit=3))
print() 