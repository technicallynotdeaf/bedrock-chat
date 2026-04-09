"""
Tavily Extract tool — extracts clean content from one or more URLs.

Uses the Tavily Extract API to pull structured content from web pages.
Useful when the model needs to read the full content of specific URLs
(e.g., from search results, user-provided links, or citations).
"""

import logging
from typing import Union

from app.repositories.models.custom_bot import BotModel
from strands import tool
from strands.types.tools import AgentTool as StrandsAgentTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Max characters per extracted page to avoid flooding the context window.
MAX_CONTENT_CHARS = 8000


def _truncate(text: str, limit: int = MAX_CONTENT_CHARS) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[Truncated — {len(text)} characters total]"


def create_tavily_extract_tool(bot: BotModel | None = None) -> StrandsAgentTool:
    @tool
    def tavily_extract(urls: list[str]) -> dict:
        """
        Extract the full content of one or more web pages.

        Use this tool when you need to read the complete content of specific URLs,
        such as articles, documentation pages, or any web page the user has referenced.
        Returns clean, structured content (markdown) extracted from each URL.

        Args:
            urls: A list of URLs to extract content from (up to 20 URLs).

        Returns:
            dict: ToolResult with extracted content for each URL.
        """
        from app.strands_integration.tools.internet_search import TAVILY_API_KEY

        logger.info(f"[TAVILY_EXTRACT] Extracting content from {len(urls)} URL(s)")

        if not TAVILY_API_KEY:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "Tavily API key is not configured. "
                        "Set TAVILY_API_KEY or TAVILY_API_KEY_SECRET_ARN."
                    }
                ],
            }

        if not urls:
            return {
                "status": "error",
                "content": [{"text": "No URLs provided."}],
            }

        # Limit to 20 URLs (Tavily API limit)
        if len(urls) > 20:
            urls = urls[:20]
            logger.warning("[TAVILY_EXTRACT] Truncated URL list to 20 (API limit)")

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.extract(
                urls=urls,
                format="markdown",
                timeout=30,
            )

            results = response.get("results", [])
            failed = response.get("failed_results", [])

            if failed:
                logger.warning(f"[TAVILY_EXTRACT] Failed URLs: {failed}")

            content_blocks: list[dict] = []
            for r in results:
                raw_content = r.get("raw_content", "")
                url = r.get("url", "")
                content_blocks.append(
                    {
                        "json": {
                            "content": _truncate(raw_content),
                            "source_name": url,
                            "source_link": url,
                        }
                    }
                )

            # Report any failures
            for f in failed:
                fail_url = f.get("url", "unknown")
                fail_error = f.get("error", "unknown error")
                content_blocks.append(
                    {
                        "json": {
                            "content": f"Failed to extract: {fail_error}",
                            "source_name": f"[FAILED] {fail_url}",
                            "source_link": fail_url,
                        }
                    }
                )

            logger.info(
                f"[TAVILY_EXTRACT] Extracted {len(results)} page(s), "
                f"{len(failed)} failed"
            )

            return {
                "status": "success" if results else "error",
                "content": content_blocks
                if content_blocks
                else [{"text": "No content could be extracted from the provided URLs."}],
            }

        except Exception as e:
            logger.error(f"[TAVILY_EXTRACT] Error: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Extract error: {str(e)}"}],
            }

    return tavily_extract
