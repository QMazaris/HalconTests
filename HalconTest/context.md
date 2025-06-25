# HALCON MCP Server Context

## ⚠️ Key Rules
**Never guess HALCON operator syntax. Always validate using the MCP server tools.**
**Never invent HALCON functions, operators, or mathematical expressions. Only use validated HALCON operators.**
**Do not write inline comments.**

## 🛠️ Available MCP Tools

### Primary Tool: `get_halcon_operator()`
**Use this for all code generation and validation tasks.**

```python
get_halcon_operator(name="threshold", detail="info")
```

**Detail levels:**
- `"signature"` - Just the function signature/syntax
- `"info"` - Name, signature, description, and URL (default)
- `"full"` - Complete documentation with page dump

### Search Tool: `search_halcon_operators()`
**Use when you don't know the exact operator name.**

```python
search_halcon_operators(query="image threshold", limit=10)
```

### Browse Tool: `list_halcon_operators()`
**Use for discovering operators with pagination.**

```python
list_halcon_operators(offset=0, limit=50)
```

## 📦 MCP Resource
Access the operators database directly:
```python
read_resource("halcon://operators")
```

## ✅ Best Practices

1. **Always use `get_halcon_operator()` for known operator names**
2. **Use `search_halcon_operators()` when unsure of exact names**
3. **Never invent HALCON syntax, functions, or expressions - always validate first**
4. **If you need a mathematical operation, search for the appropriate HALCON operator**
5. **Start with `detail="info"` for most cases**
6. **Use `detail="signature"` when you only need syntax**
7. **Use `detail="full"` only when complete documentation is needed**

## ⚠️ Data Shape & Indexing Rule
When sampling or selecting points from two or more data sources (e.g., contours, tuples), always generate index tuples for each source independently, scaled to their respective lengths. Never use indices from one data source to access another unless their lengths are guaranteed to be identical.

## 🎯 Workflow

1. **Known operator**: `get_halcon_operator(name="operator_name")`
2. **Unknown operator**: `search_halcon_operators(query="your search terms")`
3. **Need mathematical operation**: Search for HALCON operators that perform the calculation
4. **Generate code**: Use only the returned signature and description from MCP server
5. **Validate**: Double-check that all syntax comes from MCP server responses