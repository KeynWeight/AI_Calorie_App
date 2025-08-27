# Food Data Central MCP Server

This is a Model Context Protocol (MCP) server for exposing API access to
the [USDA's FoodData Central API](https://fdc.nal.usda.gov/api-guide).

## Features

- Search for foods in the USDA FoodData Central database
- Access food nutrient information
- Paginated results
- Support for multiple data types (Foundation, SR Legacy, Survey, Branded)

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Build the project:
   ```bash
   npm run build
   ```

## Running the Server

The server uses stdio transport, which means it's designed to be run as a subprocess by an MCP client. To run it directly:

```bash
# Set the USDA API key as an environment variable
export USDA_API_KEY=your-api-key-here
npm start
```

For development with hot reloading:

```bash
# Set the USDA API key as an environment variable
export USDA_API_KEY=your-api-key-here
npm run dev
```

## Using with Claude Desktop

To use this MCP server with Claude Desktop:

1. Open the Claude Desktop settings:

   - On macOS: Click on the Claude menu and select "Settings..."
   - On Windows: Click on the Claude menu and select "Settings..."

2. In the Settings pane, click on "Developer" in the left-hand bar, and then click on "Edit Config"

3. This will create or open a configuration file at:

   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

4. Add the Food Data Central MCP server to the configuration file:

   ```json
   {
     "mcpServers": {
       "food-data-central": {
         "command": "npx",
         "args": ["tsx", "/path/to/food-data-central-mcp-server/src/index.ts"],
         "env": {
           "PATH": "/opt/homebrew/bin",
           "USDA_API_KEY": "<INSERT KEY HERE>"
         }
       }
     }
   }
   ```

   Replace `/path/to/food-data-central-mcp-server` with the absolute path to this repository, and `<INSERT KEY HERE>` with your actual USDA API key.

   Note: If you're on Windows, you may need to adjust the PATH value to include your npm global installation directory.

5. Save the configuration file and restart Claude Desktop

6. After restarting, you should see a hammer icon in the bottom right corner of the input box. Click on it to see the available tools.

Now Claude will be able to access the Food Data Central API through this MCP server. You can ask Claude to search for foods, get nutrient information, or retrieve detailed food data.

## MCP Resources and Tools

### Resources

- `food://details` - Get detailed information about a specific food by ID

  - Query parameters:
    - `fdcId`: Food Data Central ID (required)
    - `format`: Optional. 'abridged' for an abridged set of elements, 'full' for all elements (default)
    - `nutrients`: Optional. List of up to 25 nutrient numbers (comma-separated)

- `food://foods` - Get details for multiple food items using input FDC IDs

  - Query parameters:
    - `fdcIds`: List of multiple FDC IDs (required, comma-separated)
    - `format`: Optional. 'abridged' for an abridged set of elements, 'full' for all elements (default)
    - `nutrients`: Optional. List of up to 25 nutrient numbers (comma-separated)

- `food://list` - Get a paged list of foods
  - Query parameters:
    - `dataType`: Optional. Filter on a specific data type (comma-separated list)
    - `pageSize`: Optional. Maximum number of results to return (default: 50)
    - `pageNumber`: Optional. Page number to retrieve (default: 1)
    - `sortBy`: Optional. Field to sort by
    - `sortOrder`: Optional. Sort order, "asc" or "desc"

### Tools

- `search-foods` - Search for foods using keywords
  - Parameters:
    - `query`: Search terms to find foods (required)
    - `dataType`: Optional. Filter on a specific data type (array of strings)
    - `pageSize`: Optional. Maximum number of results to return (default: 50)
    - `pageNumber`: Optional. Page number to retrieve (default: 1)
    - `sortBy`: Optional. Field to sort by
    - `sortOrder`: Optional. Sort order, "asc" or "desc"
    - `brandOwner`: Optional. Filter results based on the brand owner of the food (only for Branded Foods)
    - `tradeChannel`: Optional. Filter foods containing any of the specified trade channels
    - `startDate`: Optional. Filter foods published on or after this date (format: YYYY-MM-DD)
    - `endDate`: Optional. Filter foods published on or before this date (format: YYYY-MM-DD)

## Example Usage

Get food details using the MCP resource:

```
food://details?fdcId=2345678&format=full
```

Get multiple foods using the MCP resource:

```
food://foods?fdcIds=534358,373052,616350
```

Get a list of foods using the MCP resource:

```
food://list?dataType=Foundation,SR Legacy&pageSize=10&pageNumber=1
```

Search for foods using the MCP tool:

```json
{
  "name": "search-foods",
  "arguments": {
    "query": "apple",
    "dataType": ["Foundation", "SR Legacy"],
    "pageSize": 10,
    "pageNumber": 1
  }
}
```
