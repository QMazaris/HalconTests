## ⚠️ Key Rule
**Do not guess HALCON operator syntax. Always validate against the MCP server.**
Never write inline comments

## 🔧 MCP Server Overview

The HALCON MCP server exposes the following tools and endpoints for operator lookup and validation:

### 🎯 If you know the operator name
Use:
```python
get_halcon_operator_info(name="threshold")
This will return:

Name

Signature

Description

Documentation URL

Use this for all code generation tasks.

🧰 Available Tools
Tool	When to Use	URI / Call
get_halcon_operator_info(name)	✅ Use this by default for known functions	halcon://essentials/{name}
get_halcon_operator_signature(name)	When you only need the signature line	
get_halcon_operator_page_dump(name)	When full doc is needed (rarely)	
search_halcon_operators(query)	When the function name is unknown or vague	
list_halcon_operators(offset, limit)	For browsing all operators (debug or discovery)	
get_operators_info()	For general metadata about the operator DB	

✅ Best Practice Summary
Use get_halcon_operator_info() for generating or validating code.

Never invent syntax — call the MCP server.

Fuzzy search via search_halcon_operators() if unsure of the exact name.

📦 MCP Resource URI Structure
For direct access to a function:
read_resource("halcon://essentials/{name}")

For full search fallback:
read_resource("halcon://operators")