"""
Tavily Crawl tool — crawls a website starting from a URL.

Uses the Tavily Crawl API to navigate a website, follow links, and extract
content from nested pages. Falls back to direct HTTP fetch of the starting
URL when Tavily is unavailable or fails.
"""

import logging
import re

import requests
from app.repositories.models.custom_bot import BotModel
from strands import tool
from strands.types.tools import AgentTool as StrandsAgentTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MAX_CONTENT_CHARS = 6000  # Per page — keep tighter since crawl returns many pages
REQUEST_TIMEOUT_SECONDS = 15


def _truncate(text: str, limit: int = MAX_CONTENT_CHARS) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[Truncated — {len(text)} characters total]"


def _strip_html_tags(html: str) -> str:
    """Lightweight HTML -> plain-text conversion."""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr|blockquote)[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", "", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r"[ \t]+", " ", html)
    return html.strip()


def _fetch_url_direct(url: str) -> str | None:
    """Fetch a URL via direct HTTP as a fallback when Tavily is unavailable."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; BedrockChatAgent/1.0; "
                "+https://github.com/aws-samples/bedrock-claude-chat)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, allow_redirects=True
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type:
            text = response.text
        elif "html" in content_type or "xml" in content_type:
            text = _strip_html_tags(response.text)
        else:
            text = response.text
        return text
    except Exception as e:
        logger.warning(f"[TAVILY_CRAWL] Direct fetch failed for {url}: {e}")
        return None


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

        if not url.startswith(("http://", "https://")):
            return {
                "status": "error",
                "content": [{"text": "URL must start with http:// or https://."}],
            }

        # Clamp parameters
        max_depth = max(1, min(max_depth, 5))
        limit = max(1, min(limit, 50))

        content_blocks: list[dict] = []

        # Try Tavily first if API key is available
        if TAVILY_API_KEY:
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

                # Report failures (don't include as content blocks — they add noise)
                if failed:
                    for f in failed:
                        fail_url = f.get("url", "unknown")
                        fail_error = f.get("error", "unknown error")
                        logger.warning(
                            f"[TAVILY_CRAWL] Page failed: {fail_url} — {fail_error}"
                        )

                logger.info(
                    f"[TAVILY_CRAWL] Crawled {len(results)} page(s) from {url}, "
                    f"{len(failed or [])} failed"
                )

            except Exception as e:
                logger.warning(
                    f"[TAVILY_CRAWL] Tavily API error, falling back to direct HTTP: {e}"
                )
                # Fall through to direct HTTP fallback below

        # Direct HTTP fallback — fetch starting URL if Tavily failed or was unavailable
        if not content_blocks:
            logger.info(
                f"[TAVILY_CRAWL] Using direct HTTP fallback for starting URL: {url}"
            )
            text = _fetch_url_direct(url)
            if text:
                content_blocks.append(
                    {
                        "json": {
                            "content": _truncate(text),
                            "source_name": url,
                            "source_link": url,
                        }
                    }
                )
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Could not crawl {url}. "
                            "The site may be blocking automated access. "
                            "Try using fetch_website instead for individual pages."
                        }
                    ],
                }

        return {
            "status": "success" if content_blocks else "error",
            "content": content_blocks
            if content_blocks
            else [{"text": f"No content could be crawled from {url}."}],
        }

    return tavily_crawl
