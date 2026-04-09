"""
Tavily Crawl tool — crawls a website starting from a URL.

Uses the Tavily Crawl API to navigate a website, follow links, and extract
content from nested pages. Useful for site-wide research, documentation
reading, or gathering information across multiple related pages.
"""

import logging

from app.repositories.models.custom_bot import BotModel
from strands import tool
from strands.types.tools import AgentTool as StrandsAgentTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MAX_CONTENT_CHARS = 6000  # Per page — keep tighter since crawl returns many pages


def _truncate(text: str, limit: int = MAX_CONTENT_CHARS) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[Truncated — {len(text)} characters total]"


def create_tavily_crawl_tool(bot: BotModel | None = None) -> StrandsAgentTool:
    @tool
    def tavily_crawl(
        url: str,
        max_depth: int = 2,
        limit: int = 10,
        instructions: str = "",
    ) -> dict:
        """
        Crawl a website starting from a URL, following links to discover and extract content from multiple pages.

        Use this tool when you need to explore an entire website or section of a site,
        such as reading documentation across multiple pages, gathering information
        from a blog, or researching all pages under a domain.

        Args:
            url: The starting URL to crawl from (must be http:// or https://).
            max_depth: Maximum link depth to follow from the starting URL (default 2, max 5).
            limit: Maximum number of pages to crawl (default 10, max 50).
            instructions: Optional natural language instructions to guide the crawler on what content to focus on.

        Returns:
            dict: ToolResult with extracted content from crawled pages.
        """
        from app.strands_integration.tools.internet_search import TAVILY_API_KEY

        logger.info(
            f"[TAVILY_CRAWL] Crawling {url} (depth={max_depth}, limit={limit})"
        )

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

        if not url.startswith(("http://", "https://")):
            return {
                "status": "error",
                "content": [{"text": "URL must start with http:// or https://."}],
            }

        # Clamp parameters
        max_depth = max(1, min(max_depth, 5))
        limit = max(1, min(limit, 50))

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=TAVILY_API_KEY)

            crawl_kwargs: dict = {
                "url": url,
                "max_depth": max_depth,
                "limit": limit,
                "format": "markdown",
                "timeout": 120,
            }
            if instructions:
                crawl_kwargs["instructions"] = instructions

            response = client.crawl(**crawl_kwargs)

            results = response.get("results", [])
            failed = response.get("failed_results", [])

            if failed:
                logger.warning(f"[TAVILY_CRAWL] Failed pages: {len(failed)}")

            content_blocks: list[dict] = []
            for r in results:
                raw_content = r.get("raw_content", "")
                page_url = r.get("url", "")
                content_blocks.append(
                    {
                        "json": {
                            "content": _truncate(raw_content),
                            "source_name": page_url,
                            "source_link": page_url,
                        }
                    }
                )

            # Report failures
            for f in (failed or []):
                fail_url = f.get("url", "unknown")
                fail_error = f.get("error", "unknown error")
                content_blocks.append(
                    {
                        "json": {
                            "content": f"Failed to crawl: {fail_error}",
                            "source_name": f"[FAILED] {fail_url}",
                            "source_link": fail_url,
                        }
                    }
                )

            logger.info(
                f"[TAVILY_CRAWL] Crawled {len(results)} page(s) from {url}, "
                f"{len(failed or [])} failed"
            )

            return {
                "status": "success" if results else "error",
                "content": content_blocks
                if content_blocks
                else [{"text": f"No content could be crawled from {url}."}],
            }

        except Exception as e:
            logger.error(f"[TAVILY_CRAWL] Error: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Crawl error: {str(e)}"}],
            }

    return tavily_crawl
