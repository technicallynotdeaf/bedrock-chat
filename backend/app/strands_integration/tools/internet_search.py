import logging
import os

from app.repositories.models.custom_bot import BotModel
from strands import tool
from strands.types.tools import AgentTool as StrandsAgentTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _load_tavily_api_key() -> str:
    """
    Load the Tavily API key at module import time (Lambda cold-start).

    Priority:
    1. TAVILY_API_KEY env var (set directly — useful for local dev)
    2. TAVILY_API_KEY_SECRET_ARN env var → reads the key from Secrets Manager
    """
    direct_key = os.environ.get("TAVILY_API_KEY", "")
    if direct_key:
        return direct_key

    secret_arn = os.environ.get("TAVILY_API_KEY_SECRET_ARN", "")
    if not secret_arn:
        return ""

    try:
        import boto3

        region = os.environ.get("REGION", "us-east-1")
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_arn)
        key = response.get("SecretString", "").strip()
        logger.info("Loaded Tavily API key from Secrets Manager.")
        return key
    except Exception as e:
        logger.warning(f"Could not load Tavily API key from Secrets Manager: {e}")
        return ""


TAVILY_API_KEY = _load_tavily_api_key()


def _truncate_content(content: str, max_chars: int = 1500) -> str:
    """Truncate content to a reasonable size for the model context."""
    if not content:
        return content
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "..."


def _search_with_tavily_standalone(
    query: str, time_limit: str, locale: str, api_key: str
) -> list[dict[str, str]]:
    """Search using Tavily API."""
    try:
        from tavily import TavilyClient

        logger.info(f"Executing Tavily search: query={query}, time_limit={time_limit}")

        client = TavilyClient(api_key=api_key)

        # Map time_limit (d/w/m/y) to Tavily's `days` parameter
        days_map = {"d": 1, "w": 7, "m": 30, "y": 365}
        days = days_map.get(time_limit, None)

        search_kwargs: dict = {
            "query": query,
            "max_results": 10,
            "include_answer": False,
            "include_raw_content": False,
        }
        if days:
            search_kwargs["days"] = days

        response = client.search(**search_kwargs)
        results = response.get("results", [])

        formatted: list[dict[str, str]] = []
        for r in results:
            content = r.get("content", "")
            title = r.get("title", "")
            url = r.get("url", "")
            formatted.append(
                {"content": _truncate_content(content), "source_name": title, "source_link": url}
            )

        logger.info(f"Tavily search completed. Found {len(formatted)} results")
        return formatted

    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []


def create_internet_search_tool(bot: BotModel | None) -> StrandsAgentTool:
    """Create an internet search tool with bot context captured in closure."""

    @tool
    def internet_search(
        query: str, locale: str = "en-us", time_limit: str = "d"
    ) -> dict:
        """
        Search the internet for information.

        Args:
            query: The query to search for on the internet.
            locale: The country code and language code for the search. Must be `{language}-{country}` for example `jp-jp` (Japanese - Japan), `zh-cn` (Chinese - China), `en-ca` (English - Canada), `fr-ca` (French - Canada), `en-nz` (English - New Zealand), etc. If empty the default is `en-us`.
            time_limit: Retrieve only the most recent results, for example `1w` only returns the results from the last week. Units are 'd' (day), 'w' (week), 'm' (month), 'y' (year). Use empty string to retrieve all results.

        Returns:
            dict: ToolResult format with search results in json field
        """
        logger.debug(
            f"[INTERNET_SEARCH_V3] Starting search: query={query}, locale={locale}, time_limit={time_limit}"
        )

        try:
            if TAVILY_API_KEY:
                logger.debug("[INTERNET_SEARCH_V3] Running Tavily search")
                results = _search_with_tavily_standalone(
                    query, time_limit, locale, TAVILY_API_KEY
                )
                if not results:
                    logger.warning("[INTERNET_SEARCH_V3] Tavily returned no results")
            else:
                logger.warning(
                    "[INTERNET_SEARCH_V3] No search provider configured. "
                    "Set TAVILY_API_KEY or TAVILY_API_KEY_SECRET_ARN to enable internet search."
                )
                results = []

            # Download any PDFs found in search result URLs and include as documents
            from app.pdf_url_handler import download_pdfs_from_urls

            source_urls = [r["source_link"] for r in results if r.get("source_link")]
            pdf_downloads = download_pdfs_from_urls(source_urls)

            content_blocks: list[dict] = [{"json": result} for result in results]

            for filename, pdf_bytes, source_url in pdf_downloads:
                content_blocks.append({
                    "json": {
                        "content": f"[PDF document downloaded from {source_url}]",
                        "source_name": filename,
                        "source_link": source_url,
                    }
                })
                content_blocks.append({
                    "document": {
                        "format": "pdf",
                        "name": filename.replace(".pdf", "").replace(".", "")[:50],
                        "source": {"bytes": pdf_bytes},
                    }
                })
                logger.info(
                    f"[INTERNET_SEARCH_V3] Included PDF document from search result: {source_url}"
                )

            # Return in ToolResult format to prevent Strands from converting to string
            return {
                "status": "success",
                "content": content_blocks,
            }

        except Exception as e:
            logger.error(f"[INTERNET_SEARCH_V3] Internet search error: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Search error: {str(e)}"}],
            }

    return internet_search
