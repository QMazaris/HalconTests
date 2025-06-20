# HALCON MCP Server

A Model Context Protocol (MCP) server that provides AI models with access to comprehensive HALCON operator documentation and syntax information.

## Features

- **1,981 HALCON operators** with syntax and documentation
- **Fuzzy search** for finding operators by name or functionality
- **Complete operator details** including syntax, descriptions, and documentation URLs
- **MCP protocol compliance** for integration with AI tools like Cursor

## Setup

1. **Clone and install dependencies:**
```bash
git clone https://github.com/QMazaris/HalconTests.git
cd HalconTests
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **Generate the database** (if not already present):
```bash
python halcon_scraper.py
```

## Cursor Integration

### Step 1: Configure Cursor MCP

Add this configuration to your Cursor MCP settings file (`~/.cursor/mcp.json` or `%USERPROFILE%\.cursor\mcp.json`):

```json

```

**Important:** Update the paths to match your actual installation directory.

### Step 2: Restart Cursor

After updating the MCP configuration, restart Cursor for the changes to take effect.

### Step 3: Verify Connection

In Cursor chat, you should see the HALCON MCP server listed as an available tool source.

## Using from Any Project Directory

Once configured in Cursor, the HALCON MCP server will be available from any project directory. The AI can:

1. **Search for operators:**
   ```
   "Search for HALCON operators related to image thresholding"
   ```

2. **Get operator syntax:**
   ```
   "Show me the syntax for the threshold operator in HALCON"
   ```

3. **Get complete operator info:**
   ```
   "Tell me everything about the read_image operator"
   ```

4. **List operators:**
   ```
   "List some HALCON operators for image processing"
   ```

## Available MCP Tools

The server provides these tools for AI models:

- **`search_halcon_operators`** - Fuzzy search operators by name/functionality
- **`get_halcon_operator`** - Get complete operator information
- **`get_halcon_syntax`** - Get just the syntax for an operator  
- **`list_halcon_operators`** - Browse operators with pagination

## Example Usage

When working on HALCON code in any directory, you can ask Cursor:

> "I need to read an image file and apply a threshold. What HALCON operators should I use and what's their syntax?"

The AI will use the MCP server to:
1. Search for relevant operators (`read_image`, `threshold`, etc.)
2. Get their exact syntax
3. Provide you with working HALCON code

## Technical Details

- **Database:** SQLite with 1,981 operators
- **Search:** RapidFuzz for intelligent fuzzy matching
- **Protocol:** Model Context Protocol (MCP) over stdio
- **Dependencies:** Python 3.8+, mcp, rapidfuzz, sqlite3

## Files

- `mcp_server.py` - Main MCP server implementation
- `halcon_scraper.py` - Scraper to build the database
- `halcon_operators.db` - SQLite database (generated)
- `requirements.txt` - Python dependencies
- `cursor-mcp-config.json` - Sample Cursor configuration 