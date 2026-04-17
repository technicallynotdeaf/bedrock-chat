"""
Tavily Extract tool — extracts clean content from one or more URLs.

Uses the Tavily Extract API to pull structured content from web pages.
Falls back to direct HTTP fetch when Tavily is unavailable or fails.
"""

import logging
import re

import requests
from app.repositories.models.custom_bot import BotModel
from strands import tool
from strands.types.tools import AgentTool as StrandsAgentTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Max characters per extracted page to avoid flooding the context window.
MAX_CONTENT_CHARS = 8000
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
        logger.warning(f"[TAVILY_EXTRACT] Direct fetch failed for {url}: {e}")
        return None


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

        if not urls:
            return {
                "status": "error",
                "content": [{"text": "No URLs provided."}],
            }

        # Limit to 20 URLs (Tavily API limit)
        if len(urls) > 20:
            urls = urls[:20]
            logger.warning("[TAVILY_EXTRACT] Truncated URL list to 20 (API limit)")

        content_blocks: list[dict] = []

        # Try Tavily first if API key is available
        if TAVILY_API_KEY:
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

                # Track which URLs succeeded via Tavily
                succeeded_urls = {r.get("url", "") for r in results}

                # Try direct HTTP fallback for URLs that Tavily failed on
                failed_urls = [
                    f.get("url", "") for f in (failed or []) if f.get("url")
                ]
                for fail_url in failed_urls:
                    if fail_url in succeeded_urls:
                        continue
                    logger.info(
                        f"[TAVILY_EXTRACT] Trying direct HTTP fallback for {fail_url}"
                    )
                    text = _fetch_url_direct(fail_url)
                    if text:
                        content_blocks.append(
                            {
                                "json": {
                                    "content": _truncate(text),
                                    "source_name": fail_url,
                                    "source_link": fail_url,
                                }
                            }
                        )
                    else:
                        content_blocks.append(
                            {
                                "json": {
                                    "content": f"Failed to extract content from this URL.",
                                    "source_name": f"[FAILED] {fail_url}",
                                    "source_link": fail_url,
                                }
                            }
                        )

                logger.info(
                    f"[TAVILY_EXTRACT] Extracted {len(results)} page(s) via Tavily, "
                    f"{len(failed or [])} failed"
                )

            except Exception as e:
                logger.warning(
                    f"[TAVILY_EXTRACT] Tavily API error, falling back to direct HTTP: {e}"
                )
                # Fall through to direct HTTP fallback below

        # Direct HTTP fallback for all URLs if Tavily was unavailable or failed entirely
        if not content_blocks:
            logger.info(
                "[TAVILY_EXTRACT] Using direct HTTP fallback for all URLs"
            )
            for url in urls:
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
                    content_blocks.append(
                        {
                            "json": {
                                "content": f"Failed to extract content from this URL.",
                                "source_name": f"[FAILED] {url}",
                                "source_link": url,
                            }
                        }
                    )

        has_results = any(
            "json" in b
            and isinstance(b["json"], dict)
            and not b["json"].get("source_name", "").startswith("[FAILED]")
            for b in content_blocks
        )

        return {
            "status": "success" if has_results else "error",
            "content": content_blocks
            if content_blocks
            else [{"text": "No content could be extracted from the provided URLs."}],
        }

    return tavily_extract
