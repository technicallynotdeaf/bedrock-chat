"""Utility to detect PDF URLs in user messages/search results and download them as attachments."""

import io
import logging
import os
import re
from urllib.parse import unquote, urlparse

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Max PDF size to download (default 4.5 MB to stay within Converse API limits)
MAX_PDF_SIZE_BYTES = int(os.environ.get("MAX_PDF_URL_SIZE_BYTES", 4_500_000))

# Timeout for downloading PDFs (seconds)
PDF_DOWNLOAD_TIMEOUT = int(os.environ.get("PDF_DOWNLOAD_TIMEOUT", 30))

# Maximum total PDF pages to include (Bedrock ConverseStream has a 100-page limit)
MAX_TOTAL_PDF_PAGES = int(os.environ.get("MAX_TOTAL_PDF_PAGES", 80))

# Maximum number of PDFs to download per search
MAX_PDF_DOWNLOADS = int(os.environ.get("MAX_PDF_DOWNLOADS", 5))

# Regex to find URLs ending in .pdf (case-insensitive), handling optional query params
PDF_URL_PATTERN = re.compile(
    r'https?://[^\s<>"\']+\.pdf(?:\?[^\s<>"\']*)?',
    re.IGNORECASE,
)


def extract_pdf_urls(text: str) -> list[str]:
    """Extract PDF URLs from a text string."""
    return PDF_URL_PATTERN.findall(text)


def _get_filename_from_url(url: str) -> str:
    """Extract a filename from a URL, falling back to 'document.pdf'."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    basename = os.path.basename(path)
    if basename and basename.lower().endswith(".pdf"):
        return basename
    return "document.pdf"


def download_pdf(url: str) -> tuple[str, bytes] | None:
    """Download a PDF from a URL. Returns (filename, content_bytes) or None on failure."""
    try:
        logger.info(f"Downloading PDF from URL: {url}")
        response = requests.get(
            url,
            timeout=PDF_DOWNLOAD_TIMEOUT,
            headers={"User-Agent": "BedrockChat/1.0"},
            stream=True,
        )
        response.raise_for_status()

        # Check content length if available
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_PDF_SIZE_BYTES:
            logger.warning(
                f"PDF at {url} is too large ({content_length} bytes). "
                f"Max allowed: {MAX_PDF_SIZE_BYTES} bytes."
            )
            return None

        # Read content with size limit
        content = b""
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_PDF_SIZE_BYTES:
                logger.warning(
                    f"PDF at {url} exceeded max size during download. "
                    f"Max allowed: {MAX_PDF_SIZE_BYTES} bytes."
                )
                return None

        # Verify it looks like a PDF
        if not content[:5] == b"%PDF-":
            logger.warning(f"Content from {url} does not appear to be a valid PDF.")
            return None

        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition and "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[1].strip('"').strip("'")
        else:
            filename = _get_filename_from_url(url)

        logger.info(f"Successfully downloaded PDF: {filename} ({len(content)} bytes)")
        return filename, content

    except RequestException as e:
        logger.warning(f"Failed to download PDF from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading PDF from {url}: {e}")
        return None


def count_pdf_pages(content: bytes) -> int:
    """Count the number of pages in a PDF. Returns 0 on failure."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return len(reader.pages)
    except Exception as e:
        logger.warning(f"Could not count PDF pages: {e}")
        return 0


def is_pdf_url(url: str) -> bool:
    """Check if a URL points to a PDF file."""
    if not url:
        return False
    parsed = urlparse(url)
    path = unquote(parsed.path).lower()
    return path.endswith(".pdf")


def download_pdfs_from_urls(urls: list[str]) -> list[tuple[str, bytes, str]]:
    """Download PDFs from a list of URLs. Only attempts URLs that look like PDFs.

    Enforces limits on total page count (MAX_TOTAL_PDF_PAGES) and number of
    PDFs (MAX_PDF_DOWNLOADS) to stay within Bedrock's 100-page document limit.

    Returns a list of (filename, content_bytes, source_url) tuples for successful downloads.
    """
    results: list[tuple[str, bytes, str]] = []
    total_pages = 0

    for url in urls:
        if not is_pdf_url(url):
            continue

        if len(results) >= MAX_PDF_DOWNLOADS:
            logger.info(
                f"Reached max PDF download limit ({MAX_PDF_DOWNLOADS}). Skipping remaining URLs."
            )
            break

        pdf_result = download_pdf(url)
        if pdf_result is not None:
            filename, content = pdf_result
            pages = count_pdf_pages(content)
            if pages > 0 and total_pages + pages > MAX_TOTAL_PDF_PAGES:
                logger.warning(
                    f"Skipping PDF {filename} ({pages} pages) — would exceed "
                    f"page limit ({total_pages} + {pages} > {MAX_TOTAL_PDF_PAGES})."
                )
                continue
            total_pages += pages
            results.append((filename, content, url))
            logger.info(
                f"Included PDF {filename} ({pages} pages). Total pages so far: {total_pages}"
            )

    return results
