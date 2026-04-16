"""Persistent document context store for conversations.

When a user uploads a document in a conversation, the bytes are stored in the
conversation's message map for the first turn.  On subsequent turns,
``_strip_attachments_from_history`` removes the raw bytes from the history to
avoid Bedrock's 100-page limit, leaving only a ``[Document previously provided]``
placeholder — meaning the model loses all document context from turn 2 onwards.

This module fixes that by:
1. Extracting text from document attachments the first time they appear in the
   conversation history (i.e. when they are about to be stripped).
2. Caching the extracted text in S3 under
   ``conversation-docs/{conversation_id}.json`` so extraction only happens once.
3. Returning a formatted system-instruction string that re-injects the document
   content on every turn, ensuring the model always has access to it.

S3 bucket: ``LARGE_PAYLOAD_SUPPORT_BUCKET`` (already available to the Lambda).
Prefix: ``conversation-docs/`` — a separate lifecycle rule expires objects after
90 days to cap storage costs.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.repositories.models.conversation import AttachmentContentModel

logger = logging.getLogger(__name__)

_BUCKET = os.environ.get("LARGE_PAYLOAD_SUPPORT_BUCKET", "")
_PREFIX = "conversation-docs"

# Cap extracted text per document to keep system-prompt overhead reasonable.
# 100 000 chars ≈ 25 000 tokens — enough for most documents.
_MAX_CHARS_PER_DOC = 100_000


def _s3():
    import boto3

    return boto3.client("s3")


def _s3_key(conversation_id: str) -> str:
    return f"{_PREFIX}/{conversation_id}.json"


def _load_stored_context(conversation_id: str) -> list[dict]:
    """Load cached document context from S3.  Returns [] on any failure."""
    if not _BUCKET:
        return []
    try:
        resp = _s3().get_object(Bucket=_BUCKET, Key=_s3_key(conversation_id))
        return json.loads(resp["Body"].read())
    except _s3().exceptions.NoSuchKey:
        return []
    except Exception as exc:
        # A missing or unreadable object is not fatal — we just re-extract.
        logger.warning("Failed to load document context from S3: %s", exc)
        return []


def _save_stored_context(conversation_id: str, docs: list[dict]) -> None:
    """Persist document context to S3."""
    if not _BUCKET:
        return
    try:
        _s3().put_object(
            Bucket=_BUCKET,
            Key=_s3_key(conversation_id),
            Body=json.dumps(docs, ensure_ascii=False),
            ContentType="application/json",
        )
        logger.info(
            "Saved document context for conversation %s (%d doc(s))",
            conversation_id,
            len(docs),
        )
    except Exception as exc:
        logger.error("Failed to save document context to S3: %s", exc)


def _extract_text(attachment: AttachmentContentModel) -> str:
    """Extract text from an attachment, capped at _MAX_CHARS_PER_DOC."""
    from app.document_chunker import extract_text_from_document

    try:
        text = extract_text_from_document(attachment.body, attachment.file_name)
    except Exception as exc:
        logger.warning(
            "Text extraction failed for %s: %s", attachment.file_name, exc
        )
        text = ""

    if not text.strip():
        return ""

    if len(text) > _MAX_CHARS_PER_DOC:
        # Find a clean break point near the cap
        break_pos = text.rfind("\n\n", 0, _MAX_CHARS_PER_DOC)
        if break_pos < _MAX_CHARS_PER_DOC // 2:
            break_pos = text.rfind("\n", 0, _MAX_CHARS_PER_DOC)
        if break_pos < _MAX_CHARS_PER_DOC // 2:
            break_pos = _MAX_CHARS_PER_DOC
        text = text[:break_pos] + "\n\n[Document truncated — remaining content omitted to fit context.]"

    return text


def get_or_update_document_context(
    conversation_id: str,
    historical_attachments: list[AttachmentContentModel],
) -> str:
    """Return a system-instruction string containing all document context for
    the conversation, extracting and caching any new attachments as needed.

    Parameters
    ----------
    conversation_id:
        The ID of the current conversation.
    historical_attachments:
        Attachments found in *historical* user messages (i.e. messages that are
        NOT the current turn).  These are about to be stripped by
        ``_strip_attachments_from_history``.

    Returns
    -------
    str
        A formatted instruction string to inject into the system prompt, or an
        empty string if there are no stored/new documents.
    """
    if not _BUCKET:
        logger.debug("LARGE_PAYLOAD_SUPPORT_BUCKET not set — skipping doc context")
        return ""

    # Load what we already have
    stored: list[dict] = _load_stored_context(conversation_id)
    stored_names: set[str] = {d["file_name"] for d in stored}

    updated = False
    for attachment in historical_attachments:
        if attachment.file_name in stored_names:
            continue  # already cached

        logger.info(
            "Extracting text from historical attachment '%s' for conversation %s",
            attachment.file_name,
            conversation_id,
        )
        text = _extract_text(attachment)
        stored.append({"file_name": attachment.file_name, "text": text})
        stored_names.add(attachment.file_name)
        updated = True

    if updated:
        _save_stored_context(conversation_id, stored)

    if not stored:
        return ""

    # Build the system instruction
    lines = [
        "The following document(s) were uploaded by the user earlier in this "
        "conversation. They are permanent context — refer to them whenever the "
        "user asks about their content, even if the user does not re-attach them.\n"
    ]
    for doc in stored:
        lines.append(f"--- {doc['file_name']} ---")
        if doc["text"].strip():
            lines.append(doc["text"])
        else:
            lines.append(
                "[Note: Text could not be extracted from this file. "
                "It may be a scanned document or image-based PDF.]"
            )
        lines.append("")  # blank line between docs

    return "\n".join(lines)


def find_historical_attachments(
    messages: list,
) -> list[AttachmentContentModel]:
    """Return all AttachmentContentModel items from historical user messages.

    "Historical" means every user message *except* the last one (the current
    turn), because the current message's attachments are sent to Bedrock natively
    and don't need to be re-injected as text context yet.
    """
    from app.repositories.models.conversation import AttachmentContentModel

    if not messages:
        return []

    # Find the last user message index
    last_user_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    attachments: list[AttachmentContentModel] = []
    for i, msg in enumerate(messages):
        if msg.role != "user" or i == last_user_idx:
            continue
        for content in msg.content:
            if isinstance(content, AttachmentContentModel):
                attachments.append(content)

    return attachments
