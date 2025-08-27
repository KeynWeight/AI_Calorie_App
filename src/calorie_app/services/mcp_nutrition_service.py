# services/mcp_nutrition_service.py
import asyncio
import logging
import re
from typing import Dict, Optional, List, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

from calorie_app.models.nutrition import Ingredient, DynamicNutritionData, NutrientInfo
from calorie_app.utils.config import SystemLimits
from calorie_app.utils.llm_cache import cached_agent_response

try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_core.tools import Tool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseLanguageModel = None

logger = logging.getLogger(__name__)


class IngredientMatchingAgent:
    """LangChain agent for intelligent ingredient matching to USDA database."""

    def __init__(
        self,
        usda_search_function,
        llm: Optional[BaseLanguageModel] = None,
        max_iterations: int = SystemLimits.MAX_AGENT_ITERATIONS,
        max_execution_time: float = SystemLimits.AGENT_TIMEOUT_SECONDS,
        enable_caching: bool = True,
    ):
        """
        Initialize the ingredient matching agent.

        Args:
            usda_search_function: Function to search USDA database
            llm: Language model for the agent (optional)
            max_iterations: Maximum LLM calls per search (default: 3)
            max_execution_time: Maximum time per search in seconds (default: 30)
            enable_caching: Whether to cache successful mappings (default: True)
        """
        self.usda_search = usda_search_function
        self.llm = llm
        self.agent_executor = None
        self.search_history = {}
        self.successful_mappings = {} if enable_caching else None
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        self.enable_caching = enable_caching

        # Rate limiting tracking
        self.api_calls_made = 0
        self.last_reset_time = (
            asyncio.get_event_loop().time() if hasattr(asyncio, "get_event_loop") else 0
        )
        self.calls_per_minute_limit = (
            SystemLimits.RATE_LIMIT_PER_MINUTE
        )  # Conservative limit

        if LANGCHAIN_AVAILABLE and llm:
            self._setup_agent()

    def _setup_agent(self):
        """Setup the LangChain agent with tools and prompt."""
        try:
            # Create tools for the agent
            tools = [
                Tool(
                    name="search_usda_database",
                    description="Search USDA food database with a query string. Returns list of food items with descriptions and IDs. Try multiple variations if first search doesn't work well.",
                    func=self._search_usda_wrapper,
                ),
                Tool(
                    name="analyze_ingredient_name",
                    description="Analyze complex ingredient name to extract core food components, removing cooking methods and descriptors.",
                    func=self._analyze_ingredient_name,
                ),
            ]

            # Create agent prompt with required variables
            from langchain_core.prompts import PromptTemplate

            prompt = PromptTemplate.from_template("""
You are an expert food nutrition assistant specialized in matching ingredient names to USDA food database entries.

Your task: Find the best USDA database match for the ingredient: "{ingredient_name}"

TOOLS:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: BEST_MATCH: [food_id]|[description]|[confidence_score]

Dynamic Strategy - No hardcoded mappings, discover everything through intelligent searching:

1. ANALYZE: Use analyze_ingredient_name to understand the core food
2. SEARCH ORIGINAL: Try the original ingredient name first
3. SEARCH SIMPLIFIED: If no good results, try the simplified core food name
4. SEARCH VARIATIONS: Try intelligent variations like:
   - Individual words from the ingredient
   - Plural/singular forms
   - Common food category terms that you discover from search results
   - Alternative names you discover from search result descriptions
5. EVALUATE: Compare all search results and pick the best match
6. LEARN: Use search results to discover new variations dynamically

Examples of dynamic discovery:
- "grilled chicken breast" → analyze → core: "chicken breast" → search → see "broilers or fryers" in results → learn that "broiler" is chicken
- "red bell peppers" → search → see "capsicum" in results → learn that peppers = capsicum
- "cooking oil" → search "oil" → see "vegetable oil", "canola oil" in results → learn common oil types

BE CREATIVE and use search results to discover food relationships dynamically!

Begin!

Question: Find the best USDA database match for the ingredient: "{ingredient_name}"
Thought: {agent_scratchpad}
""")

            # Create agent
            if LANGCHAIN_AVAILABLE:
                agent = create_react_agent(self.llm, tools, prompt)
                self.agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=False,  # Reduce verbosity to save tokens
                    max_iterations=self.max_iterations,
                    max_execution_time=self.max_execution_time,
                    return_intermediate_steps=True,
                    handle_parsing_errors=True,  # Handle parsing errors gracefully
                    early_stopping_method="force",  # Use force instead of generate
                )

                logger.info("Ingredient matching agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to setup agent: {str(e)}")
            self.agent_executor = None

    def _search_usda_wrapper(self, query: str) -> str:
        """Wrapper for USDA search function to work with agent tools."""
        try:
            # This will be called by the agent, so we need to convert async to sync
            import asyncio

            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to schedule in thread pool
                future = asyncio.run_coroutine_threadsafe(
                    self.usda_search(query, max_results=5), loop
                )
                results = future.result(timeout=30)
            except RuntimeError:
                # No running loop, we can use asyncio.run
                results = asyncio.run(self.usda_search(query, max_results=5))

            if not results:
                return f"No results found for query: {query}"

            # Format results for agent
            formatted_results = []
            for i, item in enumerate(results[:5]):  # Limit to top 5 for agent
                formatted_results.append(
                    f"{i + 1}. ID: {item.get('fdcId', 'N/A')} - {item.get('description', 'No description')}"
                )

            return "Search results:\n" + "\n".join(formatted_results)

        except Exception as e:
            return f"Search error: {str(e)}"

    def _analyze_ingredient_name(self, ingredient_name: str) -> str:
        """Analyze ingredient name to extract core food components."""
        try:
            # Remove common cooking methods and descriptors
            cooking_methods = [
                "grilled",
                "fried",
                "baked",
                "roasted",
                "steamed",
                "boiled",
                "sauteed",
                "pan-fried",
                "deep-fried",
                "broiled",
                "braised",
                "stewed",
            ]

            descriptors = [
                "fresh",
                "frozen",
                "canned",
                "dried",
                "organic",
                "free-range",
                "grass-fed",
                "raw",
                "cooked",
                "chopped",
                "diced",
                "sliced",
                "whole",
                "ground",
                "crushed",
                "extra",
                "virgin",
                "unsalted",
                "salted",
                "low-fat",
                "fat-free",
                "lean",
            ]

            # Clean the ingredient name
            words = ingredient_name.lower().split()
            core_words = []

            for word in words:
                # Remove punctuation
                clean_word = re.sub(r"[^\w\s]", "", word)

                # Skip cooking methods and descriptors
                if clean_word not in cooking_methods and clean_word not in descriptors:
                    core_words.append(clean_word)

            core_food = " ".join(core_words)

            analysis = f"""
Ingredient: {ingredient_name}
Core food identified: {core_food}
Removed terms: {[w for w in words if re.sub(r"[^\w\s]", "", w) in cooking_methods + descriptors]}
Reasoning: Extracted the essential food item by removing cooking methods and quality descriptors.
"""

            return analysis

        except Exception as e:
            return f"Analysis error: {str(e)}"

    @cached_agent_response()
    async def find_batch_matches(self, batch_prompt: str) -> Optional[str]:
        """
        Use simple LLM call for batch ingredient matching instead of complex agent.

        Args:
            batch_prompt: Complete prompt with ingredients and search results

        Returns:
            JSON formatted response for batch processing
        """
        if not self.llm:
            logger.warning("LLM not available, cannot perform batch matching")
            return None

        # Rate limiting check
        if not self._check_rate_limit():
            logger.warning("⚠️  Rate limit exceeded, skipping batch LLM matching")
            return None

        try:
            logger.info(
                f"[LLM] Using simple LLM for batch ingredient matching (API calls used: {self.api_calls_made})"
            )

            # Create a JSON-formatted prompt that matches the expected response format
            simple_prompt = f"""{batch_prompt}

CRITICAL: Return ONLY valid JSON in this exact format, no other text:

{{
  "matches": [
    {{
      "ingredient": "Fettuccine Pasta",
      "fdcId": "168874",
      "description": "Pasta, cooked, enriched",
      "confidence": "high",
      "matched": true
    }},
    {{
      "ingredient": "Lettuce",
      "matched": false,
      "reason": "Low confidence match"
    }}
  ]
}}

Use exact fdcId numbers from the search results provided above.
Only set matched=true if confidence >= 70%.
Return valid JSON only - no explanations."""

            # Direct LLM call instead of agent
            result = await self._run_simple_llm_async(simple_prompt)

            if result:
                logger.info("[SUCCESS] Simple LLM completed processing")
                return result

            logger.warning("[FAIL] Simple LLM could not process matching")
            return None

        except Exception as e:
            logger.error(f"[ERROR] Simple LLM matching failed: {str(e)}")
            return None

    async def _run_simple_llm_async(self, prompt: str) -> Optional[str]:
        """Run a simple LLM call asynchronously."""
        try:
            # Track API usage
            self._increment_api_calls()

            # Run LLM in thread pool to avoid blocking
            import asyncio

            loop = asyncio.get_event_loop()

            def run_llm():
                # Use the LLM directly for simple text generation
                if hasattr(self.llm, "invoke"):
                    response = self.llm.invoke(prompt)
                    return (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                else:
                    # Fallback for different LLM interfaces
                    return self.llm(prompt)

            result = await loop.run_in_executor(None, run_llm)

            logger.info(
                f"[COMPLETE] Simple LLM completed. Total API calls this minute: {self.api_calls_made}"
            )
            return result

        except Exception as e:
            logger.error(f"Error running simple LLM: {str(e)}")
            return None

    async def _run_batch_agent_async(self, batch_prompt: str) -> Optional[Dict]:
        """Run the agent asynchronously for batch processing with extended limits."""
        try:
            # Track API usage
            self._increment_api_calls()

            # Run agent in thread pool with extended timeout for batch processing
            loop = asyncio.get_event_loop()

            # Direct execution without double-wrapping in executors
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._run_batch_agent_sync, batch_prompt),
                timeout=60.0,  # Extended timeout for batch processing
            )

            logger.info(
                f"[COMPLETE] Batch agent completed. Total API calls this minute: {self.api_calls_made}"
            )
            return result

        except asyncio.TimeoutError:
            logger.error("Batch agent timed out after 60 seconds")
            return None
        except Exception as e:
            logger.error(f"Error running batch agent: {str(e)}")
            return None

    def _run_batch_agent_sync(self, batch_prompt: str) -> Dict:
        """Synchronous batch agent execution with custom configuration."""
        try:
            # Create a custom agent executor for batch processing with higher limits
            from langchain.agents import AgentExecutor
            from langchain_core.prompts import PromptTemplate
            from langchain.agents import create_react_agent

            # Create a specialized prompt template for batch processing
            batch_prompt_template = PromptTemplate.from_template("""
You are an expert food nutrition assistant specialized in matching ingredient names to USDA food database entries.

Your task: Process the following batch request: {ingredient_name}

TOOLS:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Your final response

Begin!

Question: Process the following batch request: {ingredient_name}
Thought: {agent_scratchpad}
""")

            # Create a new agent specifically for batch processing
            batch_agent = create_react_agent(
                self.llm, self.agent_executor.tools, batch_prompt_template
            )

            # Create temporary agent executor with extended limits for batch processing
            batch_executor = AgentExecutor(
                agent=batch_agent,
                tools=self.agent_executor.tools,
                verbose=False,
                max_iterations=6,  # Double the iterations for batch processing
                max_execution_time=50.0,  # Extended time limit
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                early_stopping_method="force",
            )

            # Use the correct variable name for the template
            return batch_executor.invoke({"ingredient_name": batch_prompt})

        except Exception as e:
            logger.error(f"Batch agent sync execution failed: {str(e)}")
            raise

    @cached_agent_response()
    async def find_best_match(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        """
        Use the agent to find the best ingredient match with rate limiting.

        Args:
            ingredient_name: Name of ingredient to search for

        Returns:
            Best matching food item from USDA database
        """
        # Check cache first
        if (
            self.enable_caching
            and self.successful_mappings
            and ingredient_name in self.successful_mappings
        ):
            logger.info(f"[CACHE] Using cached match for {ingredient_name}")
            return self.successful_mappings[ingredient_name]

        if not self.agent_executor:
            logger.warning("Agent not available, cannot perform intelligent matching")
            return None

        # Rate limiting check
        if not self._check_rate_limit():
            logger.warning(
                f"⚠️  Rate limit exceeded, skipping agent for: {ingredient_name}"
            )
            return None

        try:
            logger.info(
                f"[AGENT] Using agent to find match for: {ingredient_name} (API calls used: {self.api_calls_made})"
            )

            # Run the agent
            result = await self._run_agent_async(ingredient_name)

            if result and "output" in result:
                match = self._parse_agent_result(result["output"])

                if match:
                    # Cache successful match
                    if self.enable_caching and self.successful_mappings is not None:
                        self.successful_mappings[ingredient_name] = match
                    logger.info(
                        f"[SUCCESS] Agent found match: {match.get('description', 'Unknown')} (ID: {match.get('fdcId')})"
                    )
                    return match

            logger.warning(f"[FAIL] Agent could not find match for {ingredient_name}")
            return None

        except Exception as e:
            logger.error(f"[ERROR] Agent search failed for {ingredient_name}: {str(e)}")
            return None

    def _check_rate_limit(self) -> bool:
        """Check if we're within API rate limits."""
        current_time = (
            asyncio.get_event_loop().time() if hasattr(asyncio, "get_event_loop") else 0
        )

        # Reset counter every minute
        if current_time - self.last_reset_time > 60:
            self.api_calls_made = 0
            self.last_reset_time = current_time

        # Check if we're under limit
        if self.api_calls_made >= self.calls_per_minute_limit:
            return False

        return True

    def _increment_api_calls(self):
        """Track API usage."""
        self.api_calls_made += 1

    async def _run_agent_async(self, ingredient_name: str) -> Optional[Dict]:
        """Run the agent asynchronously with API call tracking."""
        try:
            # Track API usage
            self._increment_api_calls()

            # Run agent in thread pool to avoid blocking
            import concurrent.futures

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.agent_executor.invoke, {"ingredient_name": ingredient_name}
                )
                result = await loop.run_in_executor(None, future.result)

                logger.info(
                    f"[COMPLETE] Agent completed. Total API calls this minute: {self.api_calls_made}"
                )
                return result

        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return None

    def _parse_agent_result(self, agent_output: str) -> Optional[Dict[str, Any]]:
        """Parse the agent's output to extract the best match."""
        try:
            # Look for the BEST_MATCH pattern
            match = re.search(
                r"BEST_MATCH:\s*([^|]+)\|([^|]+)\|([^|\n]+)", agent_output
            )

            if match:
                food_id = match.group(1).strip()
                description = match.group(2).strip()
                confidence = match.group(3).strip()

                return {
                    "fdcId": food_id,
                    "description": description,
                    "confidence": confidence,
                    "agent_reasoning": agent_output,
                }

            # Fallback: try to extract food ID from the output
            id_match = re.search(r"ID:\s*(\d+)", agent_output)
            desc_match = re.search(r"-\s*([^\n]+)", agent_output)

            if id_match:
                return {
                    "fdcId": id_match.group(1),
                    "description": desc_match.group(1)
                    if desc_match
                    else "Agent selected",
                    "confidence": "medium",
                    "agent_reasoning": agent_output,
                }

            logger.warning(f"Could not parse agent output: {agent_output}")
            return None

        except Exception as e:
            logger.error(f"Error parsing agent result: {str(e)}")
            return None


class MCPNutritionService:
    """Service for accessing USDA Food Data Central via MCP server."""

    def __init__(
        self,
        mcp_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseLanguageModel] = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MCP nutrition service.

        Args:
            mcp_config: Configuration for MCP servers. If None, uses default USDA config.
            llm: Language model for intelligent ingredient matching (optional)
            agent_config: Agent configuration dict with keys:
                - max_iterations: Max LLM calls per search (default: 3)
                - max_execution_time: Max time per search (default: 30.0)
                - calls_per_minute_limit: API rate limit (default: 20)
                - enable_caching: Cache successful mappings (default: True)
        """
        self.mcp_client = None
        self.tools = []
        self.is_connected = False
        self.llm = llm
        self.agent = None
        self.use_agent = True  # Flag to enable/disable agent usage

        # Nutrition data cache - stores complete nutrition data during search
        self.nutrition_cache = {}  # Map: fdcId -> DynamicNutritionData

        # Agent configuration
        self.agent_config = agent_config or {
            "max_iterations": SystemLimits.MAX_AGENT_ITERATIONS,
            "max_execution_time": SystemLimits.AGENT_TIMEOUT_SECONDS,
            "calls_per_minute_limit": SystemLimits.RATE_LIMIT_PER_MINUTE,
            "enable_caching": True,
        }

        if mcp_config is None:
            # Default configuration for USDA Food Data Central MCP server
            mcp_config = {
                "food-data-central": {
                    "command": "npx",
                    "args": ["tsx", "./src/index.ts"],  # Path to USDA MCP server source
                    "env": {
                        "USDA_API_KEY": None  # Will be set from environment
                    },
                    "transport": "stdio",
                }
            }

        self.config = mcp_config

    async def connect(self) -> bool:
        """
        Connect to MCP servers and load tools.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to MCP nutrition services...")

            # Initialize MCP client
            self.mcp_client = MultiServerMCPClient(self.config)

            # Get available tools
            self.tools = await self.mcp_client.get_tools()

            self.is_connected = True
            logger.info(
                f"Connected to MCP services. Available tools: {len(self.tools)}"
            )

            # Log available tools for debugging
            tool_names = []
            for tool in self.tools:
                tool_names.append(tool.name)
                logger.info(
                    f"Available MCP tool: {tool.name} - {tool.description[:100]}..."
                )
            logger.info(f"All available tools: {tool_names}")

            # Initialize agent if LLM is provided
            if self.llm and LANGCHAIN_AVAILABLE:
                try:
                    self.agent = IngredientMatchingAgent(
                        self.search_food,
                        self.llm,
                        max_iterations=self.agent_config.get("max_iterations", 3),
                        max_execution_time=self.agent_config.get(
                            "max_execution_time", 30.0
                        ),
                        enable_caching=self.agent_config.get("enable_caching", True),
                    )
                    # Update rate limit
                    self.agent.calls_per_minute_limit = self.agent_config.get(
                        "calls_per_minute_limit", 20
                    )
                    logger.info(
                        f"Ingredient matching agent initialized (max_iterations: {self.agent_config.get('max_iterations', 3)}, rate_limit: {self.agent_config.get('calls_per_minute_limit', 20)}/min)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize agent: {str(e)}")
                    self.agent = None

            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP services: {str(e)}")
            self.is_connected = False
            return False

    def _get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a specific tool by name from available MCP tools."""
        if not self.tools:
            return None

        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    async def search_food(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for foods in USDA database.

        Args:
            query: Search query (e.g., "chicken breast")
            max_results: Maximum number of results to return

        Returns:
            List of food items with basic info
        """
        if not self.is_connected:
            logger.warning("MCP client not connected")
            return []

        try:
            # Find the search tool
            search_tool = self._get_tool("search-foods")
            if not search_tool:
                logger.error("search-foods tool not available")
                return []

            # Execute search
            result = await search_tool.ainvoke(
                {"query": query, "pageSize": max_results}
            )

            # Parse search results
            if isinstance(result, str):
                try:
                    import json

                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict) and "foods" in parsed_result:
                        return parsed_result["foods"][:max_results]
                    else:
                        logger.warning(
                            f"Unexpected parsed result format: {type(parsed_result)}"
                        )
                        return []
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON response from MCP server")
                    return []
            elif isinstance(result, dict) and "foods" in result:
                return result["foods"][:max_results]
            elif isinstance(result, list):
                return result[:max_results]
            else:
                logger.warning(f"Unexpected search result format: {type(result)}")
                return []

        except Exception as e:
            logger.error(f"Food search failed: {str(e)}")
            return []

    async def search_and_cache_nutrition_data(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for foods and pre-cache their complete nutrition data for faster matching.

        Args:
            query: Search query (e.g., "chicken breast")
            max_results: Maximum number of results to return

        Returns:
            List of food items with basic info (nutrition data cached separately)
        """
        if not self.is_connected:
            logger.warning("MCP client not connected")
            return []

        try:
            # Get basic search results first
            search_results = await self.search_food(query, max_results)

            if not search_results:
                return []

            logger.info(
                f"[CACHE] Pre-caching nutrition data for {len(search_results)} '{query}' results"
            )

            # Pre-fetch and cache complete nutrition data for all results
            cached_count = 0
            for result in search_results:
                fdc_id = str(result.get("fdcId", ""))
                if fdc_id and fdc_id != "N/A":
                    # Check if already cached
                    if fdc_id not in self.nutrition_cache:
                        try:
                            # Get complete nutrition data and cache it
                            food_details = await self.get_food_details(fdc_id)
                            if food_details:
                                # Convert to DynamicNutritionData and cache
                                dynamic_nutrition = (
                                    self._convert_usda_to_dynamic_nutrition(
                                        food_details,
                                        result.get("description", query),
                                        100.0,  # Cache at 100g baseline
                                        fdc_id,
                                        result.get("description", ""),
                                    )
                                )

                                if dynamic_nutrition:
                                    self.nutrition_cache[fdc_id] = dynamic_nutrition
                                    cached_count += 1
                                    logger.debug(
                                        f"[CACHE] Cached nutrition data for ID:{fdc_id}"
                                    )
                        except Exception as e:
                            logger.warning(
                                f"[CACHE] Failed to cache nutrition data for ID:{fdc_id}: {e}"
                            )
                    else:
                        logger.debug(f"[CACHE] Already cached: ID:{fdc_id}")

            logger.info(
                f"[CACHE] Successfully cached nutrition data for {cached_count}/{len(search_results)} results"
            )
            return search_results

        except Exception as e:
            logger.error(f"Search and cache failed: {str(e)}")
            return []

    async def get_food_details(self, food_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed nutrition information for a specific food.

        Args:
            food_id: USDA food database ID

        Returns:
            Detailed food information including nutrients
        """
        if not self.is_connected:
            logger.warning("MCP client not connected")
            return None

        try:
            # Use resource approach (the MCP server only provides food-details as a resource)
            logger.debug(f"Getting food details via resource for {food_id}")
            resource_uri = f"food://details?fdcId={food_id}&format=full"
            logger.debug(f"Attempting to get resource: {resource_uri}")

            # Try direct resource call with better error handling
            try:
                logger.info(
                    f"[RESOURCE] Attempting to fetch food details for ID: {food_id}"
                )
                results = await self.mcp_client.get_resources(
                    "food-data-central", uris=[resource_uri]
                )
                logger.debug(f"Got resource results: {len(results) if results else 0}")

                if results and len(results) > 0:
                    # Parse the response content
                    blob = results[0]
                    import json

                    try:
                        data = json.loads(
                            blob.text
                        )  # Use .text instead of .as_string()
                        logger.debug(f"Successfully parsed food details for {food_id}")
                        return data
                    except (json.JSONDecodeError, AttributeError) as parse_error:
                        # Try alternative access methods
                        try:
                            if hasattr(blob, "as_string"):
                                data = json.loads(blob.as_string())
                                logger.debug(
                                    f"Successfully parsed food details for {food_id} (fallback)"
                                )
                                return data
                            elif hasattr(blob, "contents") and blob.contents:
                                data = json.loads(blob.contents[0].text)
                                logger.debug(
                                    f"Successfully parsed food details for {food_id} (contents)"
                                )
                                return data
                            else:
                                logger.error(
                                    f"Failed to parse food details JSON for {food_id}: {parse_error}"
                                )
                                return None
                        except Exception as fallback_error:
                            logger.error(
                                f"All parsing attempts failed for {food_id}: {fallback_error}"
                            )
                            return None
                else:
                    logger.warning(
                        f"No results from food details resource for {food_id}"
                    )
                    return None

            except Exception as resource_error:
                import traceback

                logger.warning(
                    f"MCP resource approach failed for {food_id}: {str(resource_error)}"
                )
                logger.debug(f"Full traceback: {traceback.format_exc()}")

                # Fall back to direct USDA API call
                logger.info(f"[FALLBACK] Using direct USDA API for food ID: {food_id}")
                return await self._get_food_details_direct_api(food_id)

        except Exception as e:
            logger.error(f"Failed to get food details for {food_id}: {str(e)}")
            return None

    async def _get_food_details_direct_api(
        self, food_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get food details directly from USDA API as fallback when MCP resources fail.

        Args:
            food_id: USDA food database ID

        Returns:
            Detailed food information including nutrients
        """
        try:
            import urllib.request
            import urllib.parse
            import json
            import os

            # Get API key from environment
            api_key = os.getenv("USDA_API_KEY")
            if not api_key:
                logger.error("USDA_API_KEY not set, cannot use direct API fallback")
                return None

            # Make direct request to USDA API using urllib (sync in thread)
            url = f"https://api.nal.usda.gov/fdc/v1/food/{food_id}"
            params = urllib.parse.urlencode({"api_key": api_key, "format": "full"})
            full_url = f"{url}?{params}"

            logger.debug(f"[DIRECT API] Making request to: {full_url}")

            # Run synchronous request in thread pool
            import concurrent.futures
            import asyncio

            def _sync_request():
                try:
                    with urllib.request.urlopen(full_url) as response:  # nosec B310 - USDA API trusted source
                        if response.status == 200:
                            data = json.loads(response.read().decode())
                            logger.info(
                                f"[DIRECT API] Successfully retrieved food details for {food_id}"
                            )
                            return data
                        else:
                            logger.warning(
                                f"[DIRECT API] Failed with status {response.status} for food ID {food_id}"
                            )
                            return None
                except Exception as e:
                    logger.error(f"[DIRECT API] Request failed: {str(e)}")
                    return None

            # Execute in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, _sync_request)
                return result

        except Exception as e:
            logger.error(f"Direct API fallback failed for {food_id}: {str(e)}")
            return None

    @cached_agent_response()
    async def search_and_get_ingredient(
        self,
        ingredient_name: str,
        weight_grams: float = 100.0,
        return_dynamic: bool = True,
    ) -> Optional[Ingredient]:
        """
        Search for an ingredient and convert to Ingredient object.

        Args:
            ingredient_name: Name of ingredient to search for
            weight_grams: Weight in grams to calculate nutrition for
            return_dynamic: If True, includes complete nutrition data

        Returns:
            Ingredient object with USDA nutrition data
        """
        try:
            logger.info(f"Searching USDA database for: {ingredient_name}")

            # Intelligent ingredient matching
            best_match = await self._find_best_ingredient_match(ingredient_name)

            if not best_match:
                logger.warning(f"No USDA data found for {ingredient_name}")
                return None

            food_id = best_match.get("fdcId")
            usda_description = best_match.get("description", "")

            # Get detailed nutrition info
            food_details = await self.get_food_details(str(food_id))

            if not food_details:
                logger.warning(f"No detailed data found for {ingredient_name}")
                return None

            # Convert to dynamic nutrition data first
            dynamic_nutrition = self._convert_usda_to_dynamic_nutrition(
                food_details,
                ingredient_name,
                weight_grams,
                str(food_id),
                usda_description,
            )

            if not dynamic_nutrition:
                return None

            # Convert to legacy Ingredient model for compatibility
            ingredient = Ingredient.from_dynamic_nutrition(dynamic_nutrition)

            logger.info(f"Successfully retrieved USDA data for {ingredient_name}")
            logger.info(f"Found {len(dynamic_nutrition.nutrients)} nutrients")

            return ingredient

        except Exception as e:
            logger.error(
                f"USDA ingredient search failed for {ingredient_name}: {str(e)}"
            )
            return None

    async def get_complete_nutrition_data(
        self, ingredient_name: str, weight_grams: float = 100.0
    ) -> Optional[DynamicNutritionData]:
        """
        Get complete nutrition data without legacy model limitations.

        Args:
            ingredient_name: Name of ingredient to search for
            weight_grams: Weight in grams to calculate nutrition for

        Returns:
            DynamicNutritionData with all available nutrients
        """
        try:
            logger.info(f"Getting complete nutrition data for: {ingredient_name}")

            best_match = await self._find_best_ingredient_match(ingredient_name)

            if not best_match:
                return None

            food_id = best_match.get("fdcId")
            usda_description = best_match.get("description", "")

            food_details = await self.get_food_details(str(food_id))

            if not food_details:
                return None

            return self._convert_usda_to_dynamic_nutrition(
                food_details,
                ingredient_name,
                weight_grams,
                str(food_id),
                usda_description,
            )

        except Exception as e:
            logger.error(
                f"Complete nutrition data retrieval failed for {ingredient_name}: {str(e)}"
            )
            return None

    async def get_nutrition_data_by_id(
        self,
        ingredient_name: str,
        usda_id: str,
        weight_grams: float = 100.0,
        usda_description: str = "",
    ) -> Optional[DynamicNutritionData]:
        """
        Get complete nutrition data by USDA ID directly (skip matching).

        Args:
            ingredient_name: Name of ingredient for display
            usda_id: USDA food database ID
            weight_grams: Weight in grams to calculate nutrition for
            usda_description: USDA food description

        Returns:
            DynamicNutritionData with all available nutrients
        """
        try:
            logger.info(
                f"Getting nutrition data by ID for: {ingredient_name} (ID: {usda_id})"
            )

            food_details = await self.get_food_details(usda_id)

            if not food_details:
                logger.warning(f"No food details found for ID: {usda_id}")
                return None

            return self._convert_usda_to_dynamic_nutrition(
                food_details, ingredient_name, weight_grams, usda_id, usda_description
            )

        except Exception as e:
            logger.error(
                f"Nutrition data retrieval by ID failed for {ingredient_name} (ID: {usda_id}): {str(e)}"
            )
            return None

    def get_cached_nutrition_data(
        self, fdc_id: str, ingredient_name: str, weight_grams: float
    ) -> Optional[DynamicNutritionData]:
        """
        Get complete nutrition data from cache, scaled to the requested weight.

        Args:
            fdc_id: USDA food ID
            ingredient_name: Display name for ingredient
            weight_grams: Weight to scale nutrition to

        Returns:
            DynamicNutritionData scaled to requested weight, or None if not cached
        """
        try:
            if fdc_id not in self.nutrition_cache:
                logger.debug(f"[CACHE MISS] No cached data for ID:{fdc_id}")
                return None

            # Get the cached baseline data (stored at 100g)
            baseline_nutrition = self.nutrition_cache[fdc_id]

            # Scale to requested weight
            if weight_grams == baseline_nutrition.weight_grams:
                # Same weight, return as-is but update ingredient name
                scaled_nutrition = baseline_nutrition.model_copy(deep=True)
                scaled_nutrition.ingredient_name = ingredient_name
                logger.info(
                    f"[CACHE HIT] Using cached data for ID:{fdc_id} (same weight: {weight_grams}g)"
                )
                return scaled_nutrition

            # Different weight, need to scale
            scale_factor = weight_grams / baseline_nutrition.weight_grams
            logger.info(
                f"[CACHE HIT] Scaling cached data for ID:{fdc_id} ({baseline_nutrition.weight_grams}g -> {weight_grams}g)"
            )

            # Create scaled version
            scaled_nutrients = {}
            for nutrient_id, nutrient_info in baseline_nutrition.nutrients.items():
                scaled_nutrients[nutrient_id] = NutrientInfo(
                    name=nutrient_info.name,
                    amount=round(nutrient_info.amount * scale_factor, 3),
                    unit=nutrient_info.unit,
                    nutrient_id=nutrient_info.nutrient_id,
                    amount_per_100g=nutrient_info.amount_per_100g,  # Keep original per-100g value
                )

            return DynamicNutritionData(
                ingredient_name=ingredient_name,
                weight_grams=weight_grams,
                usda_food_id=baseline_nutrition.usda_food_id,
                usda_description=baseline_nutrition.usda_description,
                nutrients=scaled_nutrients,
                raw_usda_data=baseline_nutrition.raw_usda_data,
            )

        except Exception as e:
            logger.error(
                f"Error retrieving cached nutrition data for {fdc_id}: {str(e)}"
            )
            return None

    def _convert_usda_to_dynamic_nutrition(
        self,
        usda_data: Dict[str, Any],
        ingredient_name: str,
        weight_grams: float,
        food_id: str,
        usda_description: str,
    ) -> Optional[DynamicNutritionData]:
        """
        Convert USDA food data to DynamicNutritionData with all nutrients.

        Args:
            usda_data: Raw USDA food data
            ingredient_name: Display name for ingredient
            weight_grams: Weight to scale nutrition to
            food_id: USDA food ID
            usda_description: USDA food description

        Returns:
            DynamicNutritionData with all available nutrients
        """
        try:
            # Extract nutrients from USDA data
            nutrients = usda_data.get("foodNutrients", [])

            # Scale factor for weight adjustment
            scale_factor = weight_grams / 100.0  # USDA data is per 100g

            # Process all nutrients dynamically
            nutrient_dict = {}
            for nutrient in nutrients:
                nutrient_info = nutrient.get("nutrient", {})
                nutrient_id = str(nutrient_info.get("id", ""))
                nutrient_name = nutrient_info.get("name", f"Nutrient_{nutrient_id}")
                unit = nutrient_info.get("unitName", "")
                amount_per_100g = nutrient.get("amount", 0)

                if nutrient_id and amount_per_100g is not None:
                    # Scale amount to requested weight
                    scaled_amount = float(amount_per_100g) * scale_factor

                    nutrient_dict[nutrient_id] = NutrientInfo(
                        name=nutrient_name,
                        amount=round(scaled_amount, 3),
                        unit=unit,
                        nutrient_id=nutrient_id,
                        amount_per_100g=float(amount_per_100g),
                    )

            return DynamicNutritionData(
                ingredient_name=ingredient_name,
                weight_grams=weight_grams,
                usda_food_id=food_id,
                usda_description=usda_description,
                nutrients=nutrient_dict,
                raw_usda_data=usda_data,
            )

        except Exception as e:
            logger.error(f"Failed to convert USDA data for {ingredient_name}: {str(e)}")
            return None

    @cached_agent_response()
    async def _find_best_ingredient_match(
        self, ingredient_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Intelligently find the best ingredient match using agent or rule-based strategies.

        Args:
            ingredient_name: Name of ingredient to search for

        Returns:
            Best matching food item from USDA database
        """
        try:
            # Strategy 1: Use LangChain agent if available
            if self.use_agent and self.agent and self.agent.agent_executor:
                logger.info(
                    f"Using intelligent agent for ingredient matching: {ingredient_name}"
                )
                agent_result = await self.agent.find_best_match(ingredient_name)
                if agent_result:
                    return agent_result
                else:
                    logger.info(
                        f"Agent failed, falling back to rule-based search for: {ingredient_name}"
                    )

            # Strategy 2: Fallback to rule-based approach
            return await self._rule_based_ingredient_search(ingredient_name)

        except Exception as e:
            logger.error(
                f"Error finding best ingredient match for {ingredient_name}: {str(e)}"
            )
            return None

    @cached_agent_response()
    async def _rule_based_ingredient_search(
        self, ingredient_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Rule-based ingredient search as fallback when agent is not available.

        Args:
            ingredient_name: Name of ingredient to search for

        Returns:
            Best matching food item from USDA database
        """
        try:
            logger.info(f"Using rule-based search for: {ingredient_name}")

            # Strategy 1: Direct search
            search_results = await self.search_food(
                ingredient_name, max_results=SystemLimits.MAX_SEARCH_RESULTS
            )

            if not search_results:
                # Strategy 2: Try simplified search (remove adjectives, etc.)
                simplified_name = self._simplify_ingredient_name(ingredient_name)
                if simplified_name != ingredient_name:
                    logger.info(
                        f"Trying simplified search: '{simplified_name}' for '{ingredient_name}'"
                    )
                    search_results = await self.search_food(
                        simplified_name, max_results=SystemLimits.MAX_SEARCH_RESULTS
                    )

            if not search_results:
                # Strategy 3: Try simple word variations (no hardcoded synonyms)
                words = ingredient_name.lower().split()
                if len(words) > 1:
                    # Try first word only
                    logger.info(
                        f"Trying first word: '{words[0]}' for '{ingredient_name}'"
                    )
                    search_results = await self.search_food(
                        words[0], max_results=SystemLimits.MAX_SEARCH_RESULTS
                    )

                    if not search_results and len(words) > 2:
                        # Try last word if still no results
                        logger.info(
                            f"Trying last word: '{words[-1]}' for '{ingredient_name}'"
                        )
                        search_results = await self.search_food(
                            words[-1], max_results=SystemLimits.MAX_SEARCH_RESULTS
                        )

            if not search_results:
                return None

            # Rank results by relevance
            best_match = self._rank_search_results(ingredient_name, search_results)

            logger.info(
                f"Rule-based match for '{ingredient_name}': {best_match.get('description', 'Unknown')} (ID: {best_match.get('fdcId')})"
            )

            return best_match

        except Exception as e:
            logger.error(f"Error in rule-based search for {ingredient_name}: {str(e)}")
            return None

    def _simplify_ingredient_name(self, name: str) -> str:
        """Minimal fallback - just remove punctuation and normalize whitespace."""
        # Only do the most basic cleanup - let AI agent handle everything else
        import re

        cleaned = re.sub(r"[^\w\s]", " ", name.lower())
        cleaned = " ".join(cleaned.split())  # Normalize whitespace
        return cleaned if cleaned else name

    def _rank_search_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Rank search results by relevance to the query.

        Args:
            query: Original search query
            results: List of USDA search results

        Returns:
            Best matching result
        """
        if not results:
            return None

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_results = []

        for result in results:
            description = result.get("description", "").lower()
            score = 0

            # Exact match bonus
            if query_lower == description:
                score += 100

            # Word overlap scoring
            desc_words = set(description.split())
            common_words = query_words.intersection(desc_words)
            score += len(common_words) * 10

            # Partial word matches
            for query_word in query_words:
                for desc_word in desc_words:
                    if query_word in desc_word or desc_word in query_word:
                        score += 5

            # Prefer "raw" or basic preparations
            if any(prep in description for prep in ["raw", "fresh", "uncooked"]):
                score += 3

            # Penalize highly processed foods
            if any(
                processed in description
                for processed in ["prepared", "seasoned", "flavored", "canned with"]
            ):
                score -= 2

            scored_results.append((score, result))

        # Sort by score (descending) and return best match
        scored_results.sort(key=lambda x: x[0], reverse=True)

        logger.debug(f"Top 3 matches for '{query}':")
        for i, (score, result) in enumerate(scored_results[:3]):
            logger.debug(
                f"  {i + 1}. {result.get('description', 'Unknown')} (Score: {score})"
            )

        return scored_results[0][1]

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status information about the intelligent agent."""
        return {
            "agent_available": self.agent is not None
            and self.agent.agent_executor is not None,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "llm_configured": self.llm is not None,
            "agent_enabled": self.use_agent,
            "cached_mappings": len(self.agent.successful_mappings) if self.agent else 0,
        }

    async def close(self):
        """Close MCP connections."""
        if self.mcp_client:
            try:
                # Note: MultiServerMCPClient might not have an explicit close method
                # This depends on the implementation
                logger.info("Closing MCP connections")
                self.is_connected = False
            except Exception as e:
                logger.error(f"Error closing MCP connections: {str(e)}")


class MCPNutritionManager:
    """Singleton manager for MCP nutrition service."""

    _instance = None
    _service = None

    @classmethod
    async def get_service(
        cls,
        mcp_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseLanguageModel] = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> MCPNutritionService:
        """
        Get or create MCP nutrition service instance.

        Args:
            mcp_config: MCP configuration (only used on first call)
            llm: Language model for intelligent matching (only used on first call)
            agent_config: Agent configuration (only used on first call)

        Returns:
            Connected MCP nutrition service
        """
        if cls._service is None:
            cls._service = MCPNutritionService(mcp_config, llm, agent_config)
            await cls._service.connect()

        return cls._service

    @classmethod
    async def close_service(cls):
        """Close the MCP service connection."""
        if cls._service:
            await cls._service.close()
            cls._service = None
