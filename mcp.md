Let’s get started with building our weather server! You can find the complete code for what we’ll be building here.

​
Prerequisite knowledge
This quickstart assumes you have familiarity with:

Python
LLMs like Claude
​
System requirements
Python 3.10 or higher installed.
You must use the Python MCP SDK 1.2.0 or higher.
​
Set up your environment
First, let’s install uv and set up our Python project and environment:


MacOS/Linux

Windows

Copy
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
Make sure to restart your terminal afterwards to ensure that the uv command gets picked up.

Now, let’s create and set up our project:


MacOS/Linux

Windows

Copy
# Create a new directory for our project
uv init weather
cd weather

# Create virtual environment and activate it
uv venv
.venv\Scripts\activate

# Install dependencies
uv add mcp[cli] httpx

# Create our server file
new-item weather.py
Now let’s dive into building your server.

​
Building your server
​
Importing packages and setting up the instance
Add these to the top of your weather.py:


Copy
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"
The FastMCP class uses Python type hints and docstrings to automatically generate tool definitions, making it easy to create and maintain MCP tools.

​
Helper functions
Next, let’s add our helper functions for querying and formatting the data from the National Weather Service API:


Copy
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""
​
Implementing tool execution
The tool execution handler is responsible for actually executing the logic of each tool. Let’s add it:


Copy
@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)
​
Running the server
Finally, let’s initialize and run the server:


Copy
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
Your server is complete! Run uv run weather.py to confirm that everything’s working.

Let’s now test your server from an existing MCP host, Claude for Desktop.

​
Testing your server with Claude for Desktop
Claude for Desktop is not yet available on Linux. Linux users can proceed to the Building a client tutorial to build an MCP client that connects to the server we just built.

First, make sure you have Claude for Desktop installed. You can install the latest version here. If you already have Claude for Desktop, make sure it’s updated to the latest version.

We’ll need to configure Claude for Desktop for whichever MCP servers you want to use. To do this, open your Claude for Desktop App configuration at ~/Library/Application Support/Claude/claude_desktop_config.json in a text editor. Make sure to create the file if it doesn’t exist.

For example, if you have VS Code installed:

MacOS/Linux
Windows

Copy
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
You’ll then add your servers in the mcpServers key. The MCP UI elements will only show up in Claude for Desktop if at least one server is properly configured.

In this case, we’ll add our single weather server like so:

MacOS/Linux
Windows
Python

Copy
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather",
        "run",
        "weather.py"
      ]
    }
  }
}
You may need to put the full path to the uv executable in the command field. You can get this by running which uv on MacOS/Linux or where uv on Windows.

Make sure you pass in the absolute path to your server.

This tells Claude for Desktop:

There’s an MCP server named “weather”
To launch it by running uv --directory /ABSOLUTE/PATH/TO/PARENT/FOLDER/weather run weather.py. 

Installation
Adding MCP to your python project
We recommend using uv to manage your Python projects.

If you haven't created a uv-managed project yet, create one:

uv init mcp-server-demo
cd mcp-server-demo
Then add MCP to your project dependencies:

uv add "mcp[cli]"
Alternatively, for projects using pip for dependencies:

pip install "mcp[cli]"
Running the standalone MCP development tools
To run the mcp command with uv:

uv run mcp
Quickstart
Let's create a simple MCP server that exposes a calculator tool and some data:

# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"
You can install this server in Claude Desktop and interact with it right away by running:

mcp install server.py
Alternatively, you can test it with the MCP Inspector:

mcp dev server.py
What is MCP?
The Model Context Protocol (MCP) lets you build servers that expose data and functionality to LLM applications in a secure, standardized way. Think of it like a web API, but specifically designed for LLM interactions. MCP servers can:

Expose data through Resources (think of these sort of like GET endpoints; they are used to load information into the LLM's context)
Provide functionality through Tools (sort of like POST endpoints; they are used to execute code or otherwise produce a side effect)
Define interaction patterns through Prompts (reusable templates for LLM interactions)
And more!
Core Concepts
Server
The FastMCP server is your core interface to the MCP protocol. It handles connection management, protocol compliance, and message routing:

# Add lifespan support for startup/shutdown with strong typing
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fake_database import Database  # Replace with your actual DB type

from mcp.server.fastmcp import FastMCP

# Create a named server
mcp = FastMCP("My App")

# Specify dependencies for deployment and development
mcp = FastMCP("My App", dependencies=["pandas", "numpy"])


@dataclass
class AppContext:
    db: Database


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    # Initialize on startup
    db = await Database.connect()
    try:
        yield AppContext(db=db)
    finally:
        # Cleanup on shutdown
        await db.disconnect()


# Pass lifespan to server
mcp = FastMCP("My App", lifespan=app_lifespan)


# Access type-safe lifespan context in tools
@mcp.tool()
def query_db() -> str:
    """Tool that uses initialized resources"""
    ctx = mcp.get_context()
    db = ctx.request_context.lifespan_context["db"]
    return db.query()
Resources
Resources are how you expose data to LLMs. They're similar to GET endpoints in a REST API - they provide data but shouldn't perform significant computation or have side effects:

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")


@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"


@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Dynamic user data"""
    return f"Profile data for user {user_id}"
Tools
Tools let LLMs take actions through your server. Unlike resources, tools are expected to perform computation and have side effects:

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")


@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters"""
    return weight_kg / (height_m**2)


@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text
Prompts
Prompts are reusable templates that help LLMs interact with your server effectively:

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("My App")


@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]
Images
FastMCP provides an Image class that automatically handles image data:

from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

mcp = FastMCP("My App")


@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")
Context
The Context object gives your tools and resources access to MCP capabilities:

from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("My App")


@mcp.tool()
async def long_task(files: list[str], ctx: Context) -> str:
    """Process multiple files with progress tracking"""
    for i, file in enumerate(files):
        ctx.info(f"Processing {file}")
        await ctx.report_progress(i, len(files))
        data, mime_type = await ctx.read_resource(f"file://{file}")
    return "Processing complete"