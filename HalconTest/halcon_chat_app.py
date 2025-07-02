#!/usr/bin/env python3
"""
HALCON Chat Interface - A ChatGPT-style web frontend for HALCON semantic search.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import sys

from flask import Flask, render_template, request, jsonify, send_from_directory
import markdown

# Add the current directory to Python path so we can import HalconTest
sys.path.insert(0, str(Path(__file__).parent))

# Import semantic search functions
from HalconTest import (
    semantic_match,
    semantic_code_search,
    get_halcon_operator,
    validate_database
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'halcon-chat-secret-key'

# Configure logging
logging.basicConfig(level=logging.INFO)

def format_operator_result(operator):
    """Format operator search result for chat display."""
    response = f"**{operator.get('name', 'Unknown')}**\n\n"
    
    if operator.get('signature'):
        response += f"**Signature:**\n```\n{operator['signature']}\n```\n\n"
    
    if operator.get('description'):
        response += f"**Description:**\n{operator['description']}\n\n"
    
    if operator.get('parameters'):
        response += f"**Parameters:**\n{operator['parameters']}\n\n"
    
    if operator.get('results'):
        response += f"**Results:**\n{operator['results']}\n\n"
    
    if operator.get('url'):
        response += f"[📖 View Documentation]({operator['url']})"
    
    return response

def format_code_result(code_example):
    """Format code example search result for chat display."""
    response = f"**{code_example.get('title', 'Code Example')}**\n\n"
    
    if code_example.get('description'):
        response += f"{code_example['description']}\n\n"
    
    if code_example.get('code'):
        response += f"**Code:**\n```halcon\n{code_example['code']}\n```\n\n"
    
    if code_example.get('tags'):
        response += f"**Tags:** {code_example['tags']}"
    
    return response

def process_query(query):
    """Process user query and return formatted response."""
    try:
        # Check for explicit search type commands
        original_query = query
        search_type = "auto"
        
        # Handle explicit commands
        if query.lower().startswith('/operators ') or query.lower().startswith('/op '):
            search_type = "operators"
            query = query.split(' ', 1)[1] if ' ' in query else query
        elif query.lower().startswith('/code ') or query.lower().startswith('/examples '):
            search_type = "code"
            query = query.split(' ', 1)[1] if ' ' in query else query
        elif query.lower().startswith('/both ') or query.lower().startswith('/all '):
            search_type = "both"
            query = query.split(' ', 1)[1] if ' ' in query else query
        
        # Auto-detect if no explicit command
        if search_type == "auto":
            query_lower = query.lower()
            
            # Strong indicators for code search
            code_keywords = ['example', 'code', 'sample', 'demo', 'tutorial', 'how to', 'show me', 'implementation']
            if any(keyword in query_lower for keyword in code_keywords):
                search_type = "code"
            else:
                search_type = "operators"
        
        # Perform searches based on type
        if search_type == "code":
            results = semantic_code_search(query=query, k=3)
            if results:
                response = "🔍 **Code Examples Search**\n\nHere are some relevant HALCON code examples:\n\n"
                for i, result in enumerate(results, 1):
                    response += f"## Example {i}\n\n"
                    response += format_code_result(result) + "\n\n---\n\n"
                response += "*💡 Tip: Use `/operators your query` to search operators instead*"
                return response
            else:
                return "I couldn't find any code examples matching your query. Try:\n- Using `/operators` to search operators instead\n- Different keywords or rephrasing your question"
        
        elif search_type == "both":
            # Search both operators and code
            operator_results = semantic_match(
                query=query, 
                k=2, 
                fields=["name", "signature", "description", "parameters", "results", "url"]
            )
            code_results = semantic_code_search(query=query, k=2)
            
            response = "🔍 **Combined Search Results**\n\n"
            
            if operator_results:
                response += "### 🛠️ Operators\n\n"
                for i, result in enumerate(operator_results, 1):
                    response += f"**{i}. {result.get('name', 'Operator')}**\n\n"
                    response += format_operator_result(result) + "\n\n"
                
            if code_results:
                response += "### 💻 Code Examples\n\n"
                for i, result in enumerate(code_results, 1):
                    response += f"**Example {i}**\n\n"
                    response += format_code_result(result) + "\n\n"
            
            if not operator_results and not code_results:
                response += "No results found in either operators or code examples."
            
            response += "\n---\n*💡 Tip: Use `/operators` or `/code` for focused searches*"
            return response
        
        else:  # operators search
            results = semantic_match(
                query=query, 
                k=3, 
                fields=["name", "signature", "description", "parameters", "results", "url"]
            )
            
            if results:
                response = "🔍 **Operators Search**\n\nHere are the most relevant HALCON operators:\n\n"
                for i, result in enumerate(results, 1):
                    response += f"## {i}. {result.get('name', 'Operator')}\n\n"
                    response += format_operator_result(result) + "\n\n---\n\n"
                response += "*💡 Tip: Use `/code your query` to search code examples instead*"
                return response
            else:
                return "I couldn't find any operators matching your query. Try:\n- Using `/code` to search code examples instead\n- Different keywords or rephrasing your question"
            
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"Sorry, I encountered an error while searching: {str(e)}"

@app.route('/')
def index():
    """Main chat interface."""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Process the query
        response = process_query(user_message)
        
        # Convert markdown to HTML for better display
        html_response = markdown.markdown(response, extensions=['codehilite', 'fenced_code'])
        
        return jsonify({
            'response': html_response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint."""
    try:
        validate_database()
        return jsonify({'status': 'healthy', 'database': 'connected'})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("🚀 Starting HALCON Chat Interface...")
    
    # Validate database on startup
    try:
        validate_database()
        print("✅ Database connection verified")
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        print("Please ensure the HALCON databases exist and are properly built.")
        sys.exit(1)
    
    print("🌐 Chat interface will be available at: http://localhost:5000")
    print("📚 Ask questions about HALCON operators and code examples!")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 