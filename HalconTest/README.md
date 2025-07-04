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
python tests/halcon_chat_app.py
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
  - **Unified Operator Search** - Automatically chooses exact or semantic search
  - **Unified Code Search** - Searches examples and intelligent code chunks
  - **Smart Auto-Detection** - Understands user intent from natural language
  - **Navigation Support** - Browse related code chunks with context
  - **Real-time database validation**
- **Endpoints:**
  - `search_operators()` - Unified operator search (replaces `get_halcon_operator` + `semantic_match`)
  - `search_code()` - Unified code search (replaces `semantic_code_search` + `enhanced_chunk_search`)
  - `list_halcon_operators()` - Browse operators with pagination
- **Usage:** `uv run HalconTest.py`

### 2. Web Chat Interface (`tests/halcon_chat_app.py`)
- **Purpose:** ChatGPT-style web interface for interactive queries
- **Features:**
  - Natural language queries with intelligent routing
  - Search type control with commands
  - Beautiful responsive UI
  - Real-time typing indicators
- **Requirements:** `--extra web` dependencies
- **Usage:** `python tests/halcon_chat_app.py`

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
├── HalconTest.py                     # Core MCP server with unified endpoints
├── build_semantic_indices.py        # FAISS index builder
├── databases/                        # 📁 All databases and indices organized
│   ├── combined.db                   # Operators database
│   ├── halcon_code_examplesV2.db     # Code examples database
│   ├── halcon_chunks_latest.db       # Intelligent code chunks database
│   ├── halcon_operators.faiss        # Pre-built operator embeddings
│   ├── halcon_code_examples.faiss    # Pre-built code embeddings
│   └── *.pkl                         # Metadata files for indices
├── tests/                            # 🧪 Testing and frontend components
│   ├── halcon_chat_app.py           # Web interface (requires --extra web)
│   └── test_mcp_search.py           # Interactive MCP server tester
└── utils/                            # 🔧 Database building utilities
    ├── build_halcon_db.py           # Operator database builder
    ├── chunk_scanner_cli.py         # Code example extractor
    ├── chunk_dev.py                 # Advanced chunking development
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
# Test MCP server functions with new unified endpoints
python tests/test_mcp_search.py

# Test unified operator search directly
python -c "from HalconTest import search_operators; print(search_operators('edge detection'))"

# Test unified code search directly  
python -c "from HalconTest import search_code; print(search_code('blob analysis examples'))"
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
- `HALCON_DB_PATH` - Path to operators database (default: `databases/combined.db`)
- `HALCON_CODE_DB_PATH` - Path to code examples database (default: `databases/halcon_code_examplesV2.db`)
- `HALCON_CHUNK_DB_PATH` - Path to code chunks database (default: `databases/halcon_chunks_latest.db`)
- `HALCON_EMBED_MODEL` - Sentence transformer model (default: `microsoft/codebert-base`)

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
# Ensure databases exist in new organized structure
ls -la databases/*.db

# Rebuild if needed
python utils/build_halcon_db.py
```

**"Module not found" for web interface**
```bash
# Install web dependencies
uv sync --extra web

# Run web interface from correct location
python tests/halcon_chat_app.py
```

**Slow search performance**
```bash
# Rebuild FAISS indices (now stored in databases/ folder)
python build_semantic_indices.py
```

### Getting Help
1. Check that all required databases exist in the `databases/` folder
2. Verify dependencies with `uv sync --extra web`
3. Test core functionality with `python tests/test_mcp_search.py`
4. Try the new unified endpoints: `search_operators()` and `search_code()`
5. Check logs for detailed error messages 