"""
Website fetching tool — retrieves the text/markdown content of any URL.

When a Tavily API key is available, uses Tavily's Extract API for high-quality
content extraction (clean markdown, main content isolation). Falls back to
direct HTTP fetch with regex-based HTML stripping when Tavily is not configured.
"""

import logging
import os
import re

import requests
from app.repositories.models.custom_bot import BotModel
from strands import tool
from strands.types.tools import AgentTool as StrandsAgentTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Maximum characters returned to the model to avoid flooding the context window
MAX_CONTENT_CHARS_TAVILY = 20000  # Tavily returns clean content, allow more
MAX_CONTENT_CHARS_FALLBACK = 8000  # Raw HTML stripping is noisier, keep tighter
REQUEST_TIMEOUT_SECONDS = 15


def _get_tavily_api_key() -> str:
    """Load Tavily API key from env var or Secrets Manager."""
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
        return response.get("SecretString", "").strip()
    except Exception as e:
        logger.warning(f"Could not load Tavily API key from Secrets Manager: {e}")
        return ""


# Cache at module level (Lambda cold start)
_TAVILY_API_KEY = _get_tavily_api_key()


def _extract_with_tavily(url: str) -> str | None:
    """
    Use Tavily Extract API to get clean content from a URL.
    Returns the extracted text/markdown, or None if extraction fails.
    """
    if not _TAVILY_API_KEY:
        return None

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=_TAVILY_API_KEY)
        response = client.extract(
            urls=[url],
            format="markdown",
            timeout=20,
        )

        results = response.get("results", [])
        if not results:
            failed = response.get("failed_results", [])
            logger.warning(f"[FETCH_WEBSITE] Tavily extract failed for {url}: {failed}")
            return None

        raw_content = results[0].get("raw_content", "")
        if not raw_content:
            return None

        # Truncate if necessary
        if len(raw_content) > MAX_CONTENT_CHARS_TAVILY:
            raw_content = (
                raw_content[:MAX_CONTENT_CHARS_TAVILY]
                + f"\n\n[Content truncated — {len(raw_content)} characters total]"
            )

        logger.info(
            f"[FETCH_WEBSITE] Tavily extracted {len(raw_content)} characters from {url}"
        )
        return raw_content

    except Exception as e:
        logger.warning(f"[FETCH_WEBSITE] Tavily extract error for {url}: {e}")
        return None


def _strip_html_tags(html: str) -> str:
    """Very lightweight HTML -> plain-text conversion using regex."""
    # Remove script and style blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace common block elements with newlines
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr|blockquote)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Remove remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Collapse excessive whitespace
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r"[ \t]+", " ", html)
    return html.strip()


def _fetch_direct(url: str, method: str, body: str) -> str:
    """Direct HTTP fetch with regex-based HTML stripping (fallback)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; BedrockChatAgent/1.0; "
            "+https://github.com/aws-samples/bedrock-claude-chat)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    request_kwargs: dict = {
        "url": url,
        "headers": headers,
        "timeout": REQUEST_TIMEOUT_SECONDS,
        "allow_redirects": True,
    }
    if body and method in ("POST", "PUT", "PATCH"):
        request_kwargs["data"] = body.encode("utf-8")

    response = requests.request(method, **request_kwargs)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if "json" in content_type:
        text = response.text
    elif "html" in content_type or "xml" in content_type:
        text = _strip_html_tags(response.text)
    else:
        text = response.text

    # Truncate if necessary
    if len(text) > MAX_CONTENT_CHARS_FALLBACK:
        text = text[:MAX_CONTENT_CHARS_FALLBACK] + f"\n\n[Content truncated — {len(text)} characters total]"

    logger.info(
        f"[FETCH_WEBSITE] Direct fetch retrieved {len(text)} characters from {url} "
        f"(HTTP {response.status_code})"
    )
    return text


def create_fetch_website_tool(bot: BotModel | None = None) -> StrandsAgentTool:
    @tool
    def fetch_website(url: str, method: str = "GET", body: str = "") -> str:
        """
        Fetch and analyse the content of a web page or HTTP endpoint.

        Use this tool when the user shares a URL and wants you to read, summarise,
        or analyse its content. Also useful for reading documentation, articles,
        blog posts, or API responses.

        When available, uses Tavily Extract for high-quality content extraction
        (clean markdown with main content isolation). Falls back to direct HTTP
        fetch otherwise.

        Args:
            url: The full URL to fetch (must start with http:// or https://).
            method: HTTP method to use — GET (default), POST, PUT, DELETE, HEAD, OPTIONS, PATCH.
            body: Optional request body for POST/PUT/PATCH requests (send as plain text or JSON string).

        Returns:
            str: The extracted content of the page (markdown when using Tavily, plain text otherwise), or an error message.
        """
        allowed_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
        method = method.upper()
        if method not in allowed_methods:
            return f"Error: Unsupported HTTP method '{method}'. Must be one of: {', '.join(sorted(allowed_methods))}."

        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."

        # If the URL points to a PDF, download it and return as a document
        from app.pdf_url_handler import download_pdf, is_pdf_url

        if is_pdf_url(url) and method == "GET":
            logger.info(f"[FETCH_WEBSITE] Detected PDF URL, downloading as document: {url}")
            pdf_result = download_pdf(url)
            if pdf_result is not None:
                filename, pdf_bytes = pdf_result
                return {
                    "status": "success",
                    "content": [
                        {
                            "json": {
                                "content": f"PDF document downloaded from {url}",
                                "source_name": filename,
                                "source_link": url,
                            }
                        },
                        {
                            "document": {
                                "format": "pdf",
                                "name": filename.replace(".pdf", "").replace(".", "")[:50],
                                "source": {"bytes": pdf_bytes},
                            }
                        },
                    ],
                }
            else:
                return f"Error: Could not download PDF from {url}. The file may be too large, inaccessible, or not a valid PDF."

        logger.info(f"[FETCH_WEBSITE] {method} {url}")

        # For GET requests, try Tavily extract first for high-quality content
        if method == "GET" and _TAVILY_API_KEY:
            tavily_content = _extract_with_tavily(url)
            if tavily_content:
                return {
                    "status": "success",
                    "content": [
                        {
                            "json": {
                                "content": tavily_content,
                                "source_name": url,
                                "source_link": url,
                            }
                        }
                    ],
                }
            logger.info(f"[FETCH_WEBSITE] Tavily extract failed, falling back to direct fetch for {url}")

        # Fallback: direct HTTP fetch
        try:
            text = _fetch_direct(url, method, body)
            if method == "GET":
                return {
                    "status": "success",
                    "content": [
                        {
                            "json": {
                                "content": text,
                                "source_name": url,
                                "source_link": url,
                            }
                        }
                    ],
                }
            return text

        except requests.exceptions.Timeout:
            return f"Error: Request to {url} timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
        except requests.exceptions.ConnectionError as e:
            return f"Error: Could not connect to {url} — {e}"
        except requests.exceptions.HTTPError as e:
            return f"Error: HTTP {e.response.status_code} from {url} — {e}"
        except Exception as e:
            logger.error(f"[FETCH_WEBSITE] Unexpected error: {e}")
            return f"Error fetching {url}: {e}"

    return fetch_website
