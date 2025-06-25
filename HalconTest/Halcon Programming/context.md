# HALCON MCP Server Context

## ⚠️ Key Rule
**Never guess HALCON operator syntax. Always validate using the MCP server tools.**
**Do not write inline comments.**


## 🎯 Recommended Workflow

### For Known Operators:
```
get_halcon_operator(name="read_image", fields=["signature", "description", "parameters"])
```

### For Unknown Operators:
```
1. semantic_match(query="load image from file", k=3)
2. get_halcon_operator(name="<result_from_step_1>", fields=["all"])
```

### For Code Examples:
```
semantic_code_search(query="image processing workflow", k=3)
```

## 🔧 Best Practices

1. **Start with semantic search** for unfamiliar functionality
2. **Always get full operator details** before writing code
3. **Use exact syntax** from the returned `signature` field
4. **Check parameters and results** fields for proper usage
5. **Validate with MCP server** rather than guessing syntax

## 📦 MCP Resource
Access the operators database directly:
```python
read_resource("halcon://operators")
```