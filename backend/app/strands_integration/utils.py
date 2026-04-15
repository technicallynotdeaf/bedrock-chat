"""
Strands integration utilities - Independent tool management.
"""

import logging
from typing import Dict

from app.bedrock import is_tooluse_supported
from app.repositories.models.custom_bot import BedrockAgentToolModel, BotModel
from app.routes.schemas.conversation import type_model_name
from strands.types.tools import AgentTool as StrandsAgentTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_strands_registered_tools(bot: BotModel | None = None) -> list[StrandsAgentTool]:
    """Get list of available Strands tools."""
    from app.strands_integration.tools.bedrock_agent import create_bedrock_agent_tool
    from app.strands_integration.tools.document_generation import (
        create_document_generation_tools,
    )
    from app.strands_integration.tools.fetch_website import create_fetch_website_tool
    from app.strands_integration.tools.image_generation import (
        create_image_generation_tool,
    )
    from app.strands_integration.tools.internet_search import (
        create_internet_search_tool,
    )
    from app.strands_integration.tools.s3_file_ops import create_s3_file_ops_tools
    from app.strands_integration.tools.tavily_crawl import create_tavily_crawl_tool
    from app.strands_integration.tools.tavily_extract import create_tavily_extract_tool

    tools: list[StrandsAgentTool] = []
    tools.append(create_internet_search_tool(bot))
    tools.append(create_tavily_extract_tool(bot))
    tools.append(create_tavily_crawl_tool(bot))
    tools.append(create_bedrock_agent_tool(bot))
    tools.append(create_fetch_website_tool(bot))
    tools.append(create_image_generation_tool(bot))
    tools.extend(create_s3_file_ops_tools(bot))
    tools.extend(create_document_generation_tools(bot))
    return tools


def get_strands_tools(
    bot: BotModel | None,
    model_name: type_model_name,
    enable_internet_search: bool = False,
) -> list[StrandsAgentTool]:
    """
    Get Strands tools based on bot configuration.

    Similar to agents/utils.py get_tools() but optimized for Strands.

    For normal chat (no bot), tools are not registered by default. However, when
    the user has toggled internet search on, we expose the ``internet_search``
    tool so the model can invoke it on any turn (first message or follow-up)
    with a context-aware query rather than relying on a one-shot pre-hook that
    uses the raw user text as the query.
    """
    if not is_tooluse_supported(model_name):
        logger.warning(
            f"Tool use is not supported for model {model_name}. Returning empty tool list."
        )
        return []

    # Normal chat (no bot): expose only internet_search when the user has enabled it.
    if bot is None:
        if enable_internet_search:
            from app.strands_integration.tools.internet_search import (
                create_internet_search_tool,
            )

            logger.info(
                "Normal chat with internet search enabled: exposing internet_search tool."
            )
            return [create_internet_search_tool(None)]
        return []

    # Return empty list if agent is not enabled
    if not bot.is_agent_enabled():
        return []

    registered_tools = get_strands_registered_tools(bot)
    tools: list[StrandsAgentTool] = []

    # Get tools based on bot's tool configuration
    for tool in bot.agent.tools:
        if tool.name not in [t.tool_name for t in registered_tools]:
            continue

        # Append tool by matching name
        matched_tool = next(
            (t for t in registered_tools if t.tool_name == tool.name), None
        )
        if matched_tool:
            tools.append(matched_tool)

    # Add knowledge tool if bot has knowledge base
    if bot.has_knowledge():
        from app.strands_integration.tools.knowledge_search import (
            create_knowledge_search_tool,
        )

        knowledge_tool = create_knowledge_search_tool(bot)
        tools.append(knowledge_tool)

    if len(tools) == 0:
        logger.warning("No tools configured for bot. Returning empty tool list.")
        return []

    logger.info(f"Strands tools configured for bot: {[t.tool_name for t in tools]}")
    return tools
