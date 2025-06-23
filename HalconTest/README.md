# HALCON MCP Server

A Model Context Protocol (MCP) server that provides comprehensive access to HALCON operator documentation and signatures. This server enables AI models to search, query, and retrieve detailed information about all 2,395+ HALCON operators.

## Features

- 🔍 **Fuzzy Search**: Find operators by partial names or functionality keywords
- 📚 **Multiple Information Levels**: 
  - Signature only (lightweight)
  - Essential info (name + signature + description)
  - Complete documentation (full page dumps)
- 🗃️ **Comprehensive Database**: 2,395 operators from HALCON 12 and Deep Learning add-on
- ⚡ **Fast Queries**: SQLite backend with optimized searches
- 🤖 **AI-Friendly**: Perfect docstrings and error handling for model consumption

## Quick Start

### Prerequisites

- Python 3.10+
- [UV package manager](https://docs.astral.sh/uv/) (recommended) or pip

### Option 1: Use Pre-built Database (Recommended)

If you have the `halcon_operators.db` file:

1. **Clone/Download the files**:
   ```bash
   # You need these files:
   # - HalconTest.py
   # - pyproject.toml  
   # - halcon_operators.db (the pre-built database)
   ```

2. **Install dependencies**:
   ```bash
   # With UV (recommended)
   uv sync

   # Or with pip
   pip install mcp[cli] rapidfuzz
   ```

3. **Run the server**:
   ```bash
   # With UV
   uv run python HalconTest.py

   # Or with pip
   python HalconTest.py
   ```

### Option 2: Build Database from Scratch

If you need to build the database yourself:

1. **Get all files**:
   ```bash
   # You need these files:
   # - HalconTest.py
   # - build_halcon_db.py
   # - scrapy.py
   # - dataExtractor.py
   # - pyproject.toml
   ```

2. **Install dependencies**:
   ```bash
   # With UV (recommended)
   uv add requests beautifulsoup4 lxml
   uv sync

   # Or with pip
   pip install mcp[cli] rapidfuzz requests beautifulsoup4 lxml
   ```

3. **Build the database** (takes ~15-20 minutes):
   ```bash
   # With UV
   uv run python build_halcon_db.py

   # Or with pip  
   python build_halcon_db.py
   ```

4. **Run the server**:
   ```bash
   # With UV
   uv run python HalconTest.py

   # Or with pip
   python HalconTest.py
   ```

## Available Endpoints

The MCP server provides these tools that AI models can use:

### 🔍 Search & Discovery
- **`search_halcon_operators(query, limit=10)`**
  - Fuzzy search by name or functionality
  - Example: `search_halcon_operators("morphology", 5)`

- **`list_halcon_operators(offset=0, limit=50)`**
  - Browse all operators with pagination
  - Example: `list_halcon_operators(0, 20)`

### 📖 Information Retrieval
- **`get_halcon_operator_signature(name)`**
  - Get just the function signature (lightweight)
  - Example: `get_halcon_operator_signature("read_image")`

- **`get_halcon_operator_info(name)`** 
  - Get essential info: name + signature + description
  - Example: `get_halcon_operator_info("threshold")`

- **`get_halcon_operator_page_dump(name)`**
  - Get complete documentation (comprehensive)
  - Example: `get_halcon_operator_page_dump("connection")`

## Usage Examples

### In an MCP Client (like Claude Desktop)

```json
{
  "mcpServers": {
    "halcon": {
      "command": "uv",
      "args": ["run", "python", "/path/to/HalconTest.py"],
      "cwd": "/path/to/project"
    }
  }
}
```

### Direct Python Usage (for testing)

```python
from HalconTest import search_halcon_operators, get_halcon_operator_info

# Search for image processing operators
results = search_halcon_operators("image filter", 5)
print(results)

# Get details about a specific operator  
info = get_halcon_operator_info("median_image")
print(info)
```

## Database Schema

The `halcon_operators.db` SQLite database contains:

| Column | Type | Description |
|--------|------|-------------|
| `name` | TEXT | Operator name (primary key) |
| `signature` | TEXT | Function signature/syntax |
| `description` | TEXT | Full operator description |
| `page_dump` | TEXT | Complete documentation content |
| `url` | TEXT | Official documentation URL |

## Troubleshooting

### Database Not Found
```
FileNotFoundError: Database not found: halcon_operators.db
```
**Solution**: Either get the pre-built database file or run `build_halcon_db.py` to create it.

### Import Errors
```
ModuleNotFoundError: No module named 'mcp'
```
**Solution**: Install dependencies with `uv sync` or `pip install mcp[cli] rapidfuzz`

### Server Won't Start
```
Database schema missing column: signature
```
**Solution**: You have an old database. Delete `halcon_operators.db` and rebuild with `build_halcon_db.py`

## Performance Notes

- **Database size**: ~50MB with 2,395 operators
- **Startup time**: ~1 second
- **Query speed**: <10ms for most operations
- **Memory usage**: ~20MB baseline

## Contributing

### Updating the Database

To rebuild with latest HALCON documentation:

```bash
# Remove old database
rm halcon_operators.db

# Rebuild (takes 15-20 minutes)
uv run python build_halcon_db.py -v
```

### Adding New Endpoints

1. Add function to `HalconTest.py` with `@mcp.tool()` decorator
2. Include comprehensive docstring with Args/Returns
3. Test with MCP client

## License

This project scrapes public HALCON documentation. Ensure compliance with MVTec's terms of service.

## Support

- Check that `halcon_operators.db` exists and is ~50MB
- Verify all dependencies are installed
- Test database with: `python -c "import sqlite3; print(sqlite3.connect('halcon_operators.db').execute('SELECT COUNT(*) FROM operators').fetchone())"` 