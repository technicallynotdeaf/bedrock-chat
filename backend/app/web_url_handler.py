"""Utility to detect web page URLs in user messages and extract their content.

Uses Tavily Extract API for high-quality content extraction when available,
falling back to direct HTTP fetch with HTML stripping. This runs at the chat
preprocessing level so it works in both "New Chat" and bot-based conversations,
regardless of whether agent tools are enabled.
"""

import logging
import os
import re
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Maximum characters of extracted content to inject per URL
MAX_CONTENT_CHARS = 20000

# Maximum number of URLs to process per message (avoid excessive API calls)
MAX_URLS_PER_MESSAGE = 5

# Timeout for direct HTTP fetch (seconds)
REQUEST_TIMEOUT_SECONDS = 15

# Regex to find URLs in text — matches http/https URLs, excludes trailing punctuation
URL_PATTERN = re.compile(
    r'https?://[^\s<>"\')\]]+',
    re.IGNORECASE,
)

# File extensions that are not web pages (handled elsewhere or not useful to extract)
NON_WEB_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".exe", ".dmg", ".bin", ".iso",
}


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


def _is_web_url(url: str) -> bool:
    """Check if a URL is a web page (not a binary file, PDF, image, etc.)."""
    parsed = urlparse(url)
    path = parsed.path.lower().rstrip("/")

    # Check if the path ends with a known non-web extension
    for ext in NON_WEB_EXTENSIONS:
        if path.endswith(ext):
            return False

    return True


def extract_web_urls(text: str) -> list[str]:
    """Extract web page URLs from text, excluding PDFs and binary files."""
    all_urls = URL_PATTERN.findall(text)
    # Clean trailing punctuation that may have been captured
    cleaned = []
    for url in all_urls:
        url = url.rstrip(".,;:!?")
        if _is_web_url(url) and url not in cleaned:
            cleaned.append(url)
    return cleaned[:MAX_URLS_PER_MESSAGE]


def _strip_html_tags(html: str) -> str:
    """Lightweight HTML to plain-text conversion."""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr|blockquote)[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", "", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r"[ \t]+", " ", html)
    return html.strip()


def _extract_with_tavily(url: str) -> str | None:
    """Use Tavily Extract API for clean content extraction. Returns None on failure."""
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
            logger.warning(f"[WEB_URL_HANDLER] Tavily extract failed for {url}: {failed}")
            return None

        raw_content = results[0].get("raw_content", "")
        if not raw_content:
            return None

        if len(raw_content) > MAX_CONTENT_CHARS:
            raw_content = (
                raw_content[:MAX_CONTENT_CHARS]
                + f"\n\n[Content truncated — {len(raw_content)} characters total]"
            )

        logger.info(f"[WEB_URL_HANDLER] Tavily extracted {len(raw_content)} chars from {url}")
        return raw_content

    except Exception as e:
        logger.warning(f"[WEB_URL_HANDLER] Tavily extract error for {url}: {e}")
        return None


def _fetch_direct(url: str) -> str | None:
    """Direct HTTP fetch with HTML stripping. Returns None on failure."""
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
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type:
            text = response.text
        elif "html" in content_type or "xml" in content_type:
            text = _strip_html_tags(response.text)
        else:
            text = response.text

        fallback_limit = 8000  # Tighter limit for raw-stripped content
        if len(text) > fallback_limit:
            text = text[:fallback_limit] + f"\n\n[Content truncated — {len(text)} characters total]"

        logger.info(f"[WEB_URL_HANDLER] Direct fetch got {len(text)} chars from {url}")
        return text

    except Exception as e:
        logger.warning(f"[WEB_URL_HANDLER] Direct fetch failed for {url}: {e}")
        return None


def fetch_url_content(url: str) -> tuple[str, str] | None:
    """Fetch content from a web URL. Returns (url, content) or None on failure.

    Tries Tavily Extract first, falls back to direct HTTP fetch.
    """
    # Try Tavily first
    content = _extract_with_tavily(url)
    if content:
        return url, content

    # Fallback to direct fetch
    content = _fetch_direct(url)
    if content:
        return url, content

    return None


def fetch_urls_content(urls: list[str]) -> list[tuple[str, str]]:
    """Fetch content from multiple URLs. Returns list of (url, content) tuples."""
    results: list[tuple[str, str]] = []
    for url in urls:
        result = fetch_url_content(url)
        if result:
            results.append(result)
    return results
