# inspect_mcp_client.py
import mcp.client as mc
import os

print("mcp.client __file__:", mc.__file__)
print("Contents of that directory:", os.listdir(os.path.dirname(mc.__file__)))
print("Attributes in mcp.client:", dir(mc))
