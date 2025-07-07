#!/usr/bin/env python3
"""
HALCON Chat Interface - A ChatGPT-style web frontend for HALCON semantic search.
Enhanced with search controls, chunk type selection, and navigation functionality.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import sys

from flask import Flask, render_template, request, jsonify, send_from_directory
import markdown

# Add the parent directory to Python path so we can import HalconTest
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import semantic search functions
from HalconTest import (
    search_operators,
    search_code,
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
    # Handle different possible field names
    title = code_example.get('procedure') or code_example.get('file') or 'Code Example'
    response = f"**{title}**\n\n"
    
    if code_example.get('description'):
        response += f"**Description:** {code_example['description']}\n\n"
    
    if code_example.get('file'):
        response += f"**File:** {code_example['file']}\n\n"
    
    if code_example.get('line_start') and code_example.get('line_end'):
        response += f"**Lines:** {code_example['line_start']}-{code_example['line_end']}\n\n"
    
    # Add boundary status information
    nav = code_example.get('navigation', {})
    boundary = nav.get('boundary_status', {})
    if boundary:
        position_info = f"**Position:** Chunk {boundary.get('current_position', '?')} of {boundary.get('total_chunks', '?')} in this file"
        
        # Add boundary indicators
        if boundary.get('is_first_chunk') and boundary.get('is_last_chunk'):
            position_info += " 🔒 (Only chunk in file)"
        elif boundary.get('is_first_chunk'):
            position_info += " ⬆️ (First chunk in file)"
        elif boundary.get('is_last_chunk'):
            position_info += " ⬇️ (Last chunk in file)"
        else:
            position_info += " 📄 (Middle of file)"
        
        response += f"{position_info}\n\n"
    
    if code_example.get('code'):
        # Determine the language for syntax highlighting
        lang = "halcon" if code_example.get('file', '').endswith('.hdev') else "text"
        
        # Combine injected context with code if available (for micro chunks)
        display_code = code_example['code']
        if code_example.get('injected_context'):
            display_code = code_example['injected_context'] + "\n" + code_example['code']
        
        response += f"**Code:**\n```{lang}\n{display_code}\n```\n\n"
    
    if code_example.get('tags'):
        response += f"**Tags:** {code_example['tags']}\n\n"
    
    return response

def process_query(query, search_type="auto", chunk_type="all", result_count=3, navigation=None, navigation_chunk_id=None):
    """Process user query with enhanced controls."""
    try:
        # Handle navigation requests
        if navigation and navigation_chunk_id:
            results = search_code(
                chunk_type=chunk_type,
                include_context=True,
                include_navigation=True,
                k=1,
                chunk_id=navigation_chunk_id,
                direction=navigation
            )
            
            if results and not results[0].get('error'):
                response = "🧭 **Navigation Result**\n\n"
                result = results[0]
                response += format_code_result(result)
                
                # Return with navigation data
                return response, [result] if result.get('navigation') else []
            else:
                # More helpful boundary message
                boundary_direction = "previous" if navigation == "previous" else "next"
                return f"📍 **End of File Reached**\n\nNo {boundary_direction} chunk available in this file. You've reached the {'beginning' if navigation == 'previous' else 'end'} of the file.", []
        
        # Handle regular searches based on search_type
        if search_type == "auto":
            # Use existing auto-detection logic
            query_lower = query.lower()
            code_keywords = ['example', 'code', 'sample', 'demo', 'tutorial', 'how to', 'show me', 'implementation']
            if any(keyword in query_lower for keyword in code_keywords):
                search_type = "code"
            else:
                search_type = "operators"
        
        navigation_data = []
        
        if search_type == "code":
            results = search_code(
                query=query, 
                chunk_type=chunk_type,
                include_context=True,
                include_navigation=True,
                k=result_count
            )
            
            if results and not (len(results) == 1 and results[0].get('error')):
                response = f"🔍 **Code Examples Search** ({chunk_type} chunks)\n\n"
                response += f"Here are the top {len(results)} relevant HALCON code examples:\n\n"
                
                for i, result in enumerate(results, 1):
                    if result.get('error'):
                        continue
                    response += f"## Example {i}\n\n"
                    response += format_code_result(result) + "\n\n---\n\n"
                    
                    # Collect navigation data
                    if result.get('navigation'):
                        navigation_data.append(result)
                
                # Removed verbose tip line for cleaner output
                return response, navigation_data
            else:
                return "I couldn't find any code examples matching your query. Try different keywords or switch to operators search.", []
        
        elif search_type == "both":
            # Search both operators and code
            operator_results = search_operators(
                query=query, 
                k=max(1, result_count // 2), 
                fields=["name", "signature", "description", "parameters", "results", "url"]
            )
            code_results = search_code(
                query=query, 
                chunk_type=chunk_type,
                include_context=True,
                include_navigation=True,
                k=max(1, result_count // 2)
            )
            
            response = f"🔍 **Combined Search Results** ({chunk_type} chunks)\n\n"
            
            # Handle operator results
            if isinstance(operator_results, dict):
                response += "### 🛠️ Operators (Exact Match)\n\n"
                response += format_operator_result(operator_results) + "\n\n"
            elif isinstance(operator_results, list) and operator_results:
                response += "### 🛠️ Operators\n\n"
                for i, result in enumerate(operator_results, 1):
                    response += f"**{i}. {result.get('name', 'Operator')}**\n\n"
                    response += format_operator_result(result) + "\n\n"
            
            # Handle code results
            if code_results and not (len(code_results) == 1 and code_results[0].get('error')):
                response += "### 💻 Code Examples\n\n"
                for i, result in enumerate(code_results, 1):
                    if result.get('error'):
                        continue
                    response += f"**Example {i}**\n\n"
                    response += format_code_result(result) + "\n\n"
                    
                    # Collect navigation data
                    if result.get('navigation'):
                        navigation_data.append(result)
            
            # Check if we have any results
            has_operator_results = (isinstance(operator_results, (dict, list)) and operator_results and not isinstance(operator_results, str))
            has_code_results = code_results and not (len(code_results) == 1 and code_results[0].get('error'))
            
            if not has_operator_results and not has_code_results:
                response += "No results found in either operators or code examples."
            
            # Removed verbose tip line
            return response, navigation_data
        
        else:  # operators search
            results = search_operators(
                query=query, 
                k=result_count, 
                fields=["name", "signature", "description", "parameters", "results", "url"]
            )
            
            # Handle different return types from search_operators
            if isinstance(results, str):
                return results, []
            elif isinstance(results, dict):
                response = "🔍 **Operators Search** (Exact Match)\n\nFound exact operator match:\n\n"
                response += format_operator_result(results) + "\n\n"
                # Tip removed
                return response, []
            elif isinstance(results, list) and results:
                response = f"🔍 **Operators Search**\n\nHere are the top {len(results)} relevant HALCON operators:\n\n"
                for i, result in enumerate(results, 1):
                    response += f"## {i}. {result.get('name', 'Operator')}\n\n"
                    response += format_operator_result(result) + "\n\n---\n\n"
                # Tip removed
                return response, []
            else:
                return "I couldn't find any operators matching your query. Try different keywords or switch to code examples.", []
            
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"Sorry, I encountered an error while searching: {str(e)}", []

@app.route('/')
def index():
    """Main chat interface."""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with enhanced controls."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        # Extract control parameters
        search_type = data.get('search_type', 'auto')
        chunk_type = data.get('chunk_type', 'all')
        result_count = int(data.get('result_count', 3))
        navigation = data.get('navigation')
        navigation_chunk_id = data.get('navigation_chunk_id')
        
        # Debug logging for navigation requests
        if navigation or navigation_chunk_id:
            logging.info(f"Navigation request: navigation={navigation}, navigation_chunk_id={navigation_chunk_id}, type={type(navigation_chunk_id)}")
            logging.info(f"Full request data: {data}")
        
        # Validate parameters
        if result_count < 1 or result_count > 20:
            result_count = 3
        
        # Enhanced validation for navigation requests
        if navigation:
            if navigation not in ['previous', 'next']:
                return jsonify({'error': 'Invalid navigation direction'}), 400
            if not navigation_chunk_id or not isinstance(navigation_chunk_id, int) or navigation_chunk_id <= 0:
                return jsonify({'error': f'Invalid navigation chunk ID: {navigation_chunk_id} (type: {type(navigation_chunk_id)})'}), 400
        
        if not user_message and not navigation:
            return jsonify({'error': 'Empty message'}), 400
        
        # Process the query
        response, navigation_data = process_query(
            user_message, 
            search_type=search_type,
            chunk_type=chunk_type,
            result_count=result_count,
            navigation=navigation,
            navigation_chunk_id=navigation_chunk_id
        )
        
        # Convert markdown to HTML for better display
        html_response = markdown.markdown(response, extensions=['codehilite', 'fenced_code'])
        
        return jsonify({
            'response': html_response,
            'navigation_data': navigation_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError as e:
        logging.error(f"Validation error: {e}")
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
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
    print("🚀 Starting Enhanced HALCON Chat Interface...")
    
    # Validate database on startup
    try:
        validate_database()
        print("✅ Database connection verified")
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        print("Please ensure the HALCON databases exist and are properly built.")
        sys.exit(1)
    
    print("🌐 Enhanced chat interface available at: http://localhost:5000")
    print("📚 Features:")
    print("   • Search type controls (operators/code/both)")
    print("   • Chunk type selection (full/micro/all)")
    print("   • Adjustable result count")
    print("   • Navigation through code chunks")
    print("   • Always-on context for micro chunks")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 