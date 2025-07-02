# HALCON MCP Server

A comprehensive MCP (Model Context Protocol) server providing semantic search access to HALCON machine vision operators and code examples.

## 🚀 Quick Start

### Core MCP Server
```bash
# Install core dependencies
uv sync

# Run MCP server (for Claude Desktop integration)
uv run HalconTest.py
```

### Optional Web Chat Interface
```bash
# Install web interface dependencies
uv sync --extra web

# Run web chat interface
python halcon_chat_app.py
# Open browser to: http://localhost:5000
```

## 📦 Installation Options

This project uses **optional dependencies** to keep installations lightweight:

### Core Installation (MCP Server Only)
```bash
uv sync
```
**Includes:** MCP server, semantic search, database access
**Size:** Minimal - only essential dependencies

### Web Interface Add-on
```bash
uv sync --extra web
```
**Adds:** Flask web server, markdown rendering for ChatGPT-style interface

### Development/Build Tools
```bash
uv sync --extra build
```
**Adds:** Web scraping tools, HTML parsing for building databases

### Everything
```bash
uv sync --extra "web,build"
```

## 🔧 Components

### 1. MCP Server (`HalconTest.py`)
- **Purpose:** Model Context Protocol server for Claude Desktop integration
- **Features:** 
  - Semantic search for HALCON operators
  - Code example search
  - Operator documentation lookup
  - Real-time database validation
- **Usage:** `uv run HalconTest.py`

### 2. Web Chat Interface (`halcon_chat_app.py`)
- **Purpose:** ChatGPT-style web interface for interactive queries
- **Features:**
  - Natural language queries
  - Search type control with commands
  - Beautiful responsive UI
  - Real-time typing indicators
- **Requirements:** `--extra web` dependencies
- **Usage:** `python halcon_chat_app.py`

### 3. Database Builders
- **`build_semantic_indices.py`** - Pre-builds FAISS vector indices
- **`utils/build_halcon_db.py`** - Scrapes and builds operator database
- **`utils/chunk_scanner_cli.py`** - Extracts code examples from files

## 🔍 Search Options

### Web Interface Commands
Use these prefix commands for precise search control:

- **`/operators your query`** - Search only operators
  ```
  /operators edge detection
  ```

- **`/code your query`** - Search only code examples
  ```
  /code blob analysis
  ```

- **`/both your query`** - Search both operators and code
  ```
  /both image processing
  ```

### Natural Language Detection
The system automatically detects search intent:
- **Operators:** "How do I find edges?", "What does read_image do?"
- **Code Examples:** "Show me examples", "How to implement", "Sample code"

## 🗄️ Database Structure

### Operators Database (`combined.db`)
- **Source:** Scraped from HALCON documentation
- **Content:** Operator names, signatures, descriptions, parameters, results
- **Count:** 2,700+ operators and procedures
- **Search:** Semantic embeddings with FAISS indexing

### Code Examples Database (`halcon_code_examplesV2.db`)
- **Source:** Parsed from HALCON example files
- **Content:** Titles, descriptions, code blocks, tags
- **Count:** 60,000+ examples
- **Search:** Title/description embeddings with FAISS indexing

## 📁 File Structure

```
HalconTest/
├── pyproject.toml                    # Project configuration with optional deps
├── HalconTest.py                     # Core MCP server
├── halcon_chat_app.py               # Web interface (requires --extra web)
├── build_semantic_indices.py        # FAISS index builder
├── combined.db                       # Operators database
├── halcon_code_examplesV2.db        # Code examples database
├── halcon_operators.faiss           # Pre-built operator embeddings
├── halcon_code_examples.faiss       # Pre-built code embeddings
├── templates/chat.html              # Web interface template
├── static/style.css                 # Web interface styling
├── static/script.js                 # Web interface functionality
└── utils/                           # Database building utilities
    ├── build_halcon_db.py           # Operator database builder
    ├── chunk_scanner_cli.py         # Code example extractor
    ├── scrapy.py                    # Web scraper
    └── dataExtractor.py             # HTML parser
```

## 🛠️ Development

### Building Databases
```bash
# Install build dependencies
uv sync --extra build

# Build operator database
python utils/build_halcon_db.py

# Extract code examples
python utils/chunk_scanner_cli.py --sqlite

# Build semantic indices
python build_semantic_indices.py
```

### Testing
```bash
# Test MCP server functions
python test_mcp_search.py

# Test semantic search directly
python -c "from HalconTest import semantic_match; print(semantic_match('edge detection'))"
```

## 🌐 Web Interface Features

- **Modern ChatGPT-style UI** with dark/light themes
- **Real-time search** with typing indicators
- **Markdown rendering** for formatted responses
- **Responsive design** for mobile and desktop
- **Error handling** with helpful suggestions
- **Search type indicators** showing what was searched
- **Command hints** for better user guidance

## 🔧 Configuration

### Environment Variables
- `HALCON_DB_PATH` - Path to operators database (default: `combined.db`)
- `HALCON_CODE_DB_PATH` - Path to code examples database (default: `halcon_code_examplesV2.db`)
- `HALCON_EMBED_MODEL` - Sentence transformer model (default: `all-MiniLM-L6-v2`)

### Performance Tuning
- **FAISS Quantization:** Enabled for datasets >500 vectors
- **Index Caching:** Pre-built indices loaded automatically
- **Batch Processing:** Optimized embedding computation

## 📋 Dependencies

### Core (Always Installed)
- `mcp[cli]>=1.9.4` - Model Context Protocol
- `sentence-transformers>=2.2.2` - Text embeddings
- `faiss-cpu>=1.7.4` - Vector similarity search
- `numpy>=1.23` - Numerical computing
- `rapidfuzz>=3.13.0` - Fast fuzzy string matching

### Web Interface (`--extra web`)
- `flask>=3.1.1` - Web framework
- `markdown>=3.8.2` - Markdown to HTML conversion

### Build Tools (`--extra build`)
- `requests>=2.31.0` - HTTP requests for scraping
- `beautifulsoup4>=4.12.0` - HTML parsing
- `lxml>=4.9.0` - XML/HTML parser

## 🤝 Usage Examples

### MCP Server with Claude Desktop
Add to your Claude Desktop configuration:
```json
{
  "mcpServers": {
    "halcon": {
      "command": "uv",
      "args": ["run", "HalconTest.py"],
      "cwd": "/path/to/HalconTest"
    }
  }
}
```

### Web Interface Queries
```
Natural Language:
- "How do I read an image in HALCON?"
- "Show me examples of blob analysis"
- "What parameters does sobel_amp take?"

Command-Based:
- "/operators gaussian filter"
- "/code edge detection"
- "/both morphological operations"
```

## 📄 License

This project provides educational access to HALCON documentation and examples. Ensure compliance with MVTec's licensing terms when using HALCON software and documentation.

## 🆘 Troubleshooting

### Common Issues

**"Database not found"**
```bash
# Ensure databases exist
ls -la *.db

# Rebuild if needed
python utils/build_halcon_db.py
```

**"Module not found" for web interface**
```bash
# Install web dependencies
uv sync --extra web
```

**Slow search performance**
```bash
# Rebuild FAISS indices
python build_semantic_indices.py
```

### Getting Help
1. Check that all required databases exist
2. Verify dependencies with `uv sync --extra web`
3. Test core functionality with `python test_mcp_search.py`
4. Check logs for detailed error messages 