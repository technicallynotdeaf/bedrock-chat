"""Persistent document store for conversations.

Documents uploaded in earlier turns are stored as raw bytes in S3 and
re-attached to the current user message on every subsequent turn, letting
Bedrock parse them natively — exactly as it did on turn 1.

Why raw bytes instead of extracted text
----------------------------------------
The previous approach extracted text with pypdf/python-docx and injected it
into the system prompt.  This caused two classes of hallucination:

1. Partial extraction — pypdf returns incomplete text for some PDFs (complex
   layouts, embedded fonts, etc.).  The model sees partial content and fills
   in the gaps with confabulation.
2. Failed extraction — pypdf returns "" for scanned/image-based PDFs.  The
   model sees "[Text could not be extracted]" and either confabulates or
   claims it cannot see the document.

Storing and re-injecting the raw bytes removes the extraction step entirely:
Bedrock's own parser (which is far more capable than pypdf) handles the
document on every turn, giving consistent, turn-1 quality throughout the
conversation.

S3 layout  (LARGE_PAYLOAD_SUPPORT_BUCKET, conversation-docs/ prefix)
----------------------------------------------------------------------
  {conversation_id}/index.json          — [{key, file_name}, …]
  {conversation_id}/{uuid}_{file_name}  — raw document bytes
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.repositories.models.conversation import AttachmentContentModel

logger = logging.getLogger(__name__)

_BUCKET = os.environ.get("LARGE_PAYLOAD_SUPPORT_BUCKET", "")
_PREFIX = "conversation-docs"

# Module-level cached S3 client — reused across Lambda invocations
_s3_client = None


def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3")
    return _s3_client


def _index_key(conversation_id: str) -> str:
    return f"{_PREFIX}/{conversation_id}/index.json"


def _load_index(conversation_id: str) -> list[dict]:
    if not _BUCKET:
        return []
    try:
        resp = _s3().get_object(Bucket=_BUCKET, Key=_index_key(conversation_id))
        return json.loads(resp["Body"].read())
    except Exception as exc:
        from botocore.exceptions import ClientError
        if isinstance(exc, ClientError):
            if exc.response.get("Error", {}).get("Code", "") in ("NoSuchKey", "404"):
                return []
        logger.warning("Failed to load document index from S3: %s", exc)
        return []


def _save_index(conversation_id: str, index: list[dict]) -> None:
    if not _BUCKET:
        return
    try:
        _s3().put_object(
            Bucket=_BUCKET,
            Key=_index_key(conversation_id),
            Body=json.dumps(index),
            ContentType="application/json",
        )
    except Exception as exc:
        logger.error("Failed to save document index to S3: %s", exc)


def _load_bytes(s3_key: str) -> bytes | None:
    try:
        resp = _s3().get_object(Bucket=_BUCKET, Key=s3_key)
        return resp["Body"].read()
    except Exception as exc:
        logger.warning("Failed to load document bytes from S3 (%s): %s", s3_key, exc)
        return None


def _save_bytes(s3_key: str, body: bytes) -> bool:
    try:
        _s3().put_object(Bucket=_BUCKET, Key=s3_key, Body=body)
        return True
    except Exception as exc:
        logger.error("Failed to save document bytes to S3 (%s): %s", s3_key, exc)
        return False


def find_historical_attachments(
    messages: list,
) -> list[AttachmentContentModel]:
    """Return AttachmentContentModel items from historical user messages.

    'Historical' means every user message except the last one (current turn).
    The current message's attachments are already in the Bedrock payload.
    """
    from app.repositories.models.conversation import AttachmentContentModel

    if not messages:
        return []

    last_user_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    result: list[AttachmentContentModel] = []
    for i, msg in enumerate(messages):
        if msg.role != "user" or i == last_user_idx:
            continue
        for content in msg.content:
            if isinstance(content, AttachmentContentModel):
                result.append(content)
    return result


def get_or_update_persistent_attachments(
    conversation_id: str,
    historical_attachments: list[AttachmentContentModel],
) -> list[AttachmentContentModel]:
    """Persist any new historical attachments to S3, then return all stored ones.

    The returned list is re-injected into the current user message so Bedrock
    can parse the documents natively on every turn — no text extraction needed.

    Parameters
    ----------
    conversation_id:
        Identifies the conversation in S3.
    historical_attachments:
        Attachments found in historical user messages (will be stripped from
        the Bedrock payload by _strip_attachments_from_history).  Any not yet
        cached are stored now.

    Returns
    -------
    list[AttachmentContentModel]
        All persisted documents for this conversation, ready to prepend to the
        current user message.  Empty list if nothing has been stored yet (e.g.
        on turn 1) or if LARGE_PAYLOAD_SUPPORT_BUCKET is not configured.
    """
    if not _BUCKET:
        return []

    index: list[dict] = _load_index(conversation_id)
    stored_names: set[str] = {entry["file_name"] for entry in index}

    updated = False
    for attachment in historical_attachments:
        if attachment.file_name in stored_names:
            continue

        # Use a random prefix so long/identical filenames don't collide
        safe_name = attachment.file_name[:80].replace("/", "_")
        doc_key = f"{_PREFIX}/{conversation_id}/{uuid.uuid4().hex}_{safe_name}"

        if _save_bytes(doc_key, bytes(attachment.body)):
            index.append({"key": doc_key, "file_name": attachment.file_name})
            stored_names.add(attachment.file_name)
            updated = True
            logger.info(
                "Persisted document '%s' for conversation %s (%d bytes)",
                attachment.file_name,
                conversation_id,
                len(attachment.body),
            )

    if updated:
        _save_index(conversation_id, index)

    if not index:
        return []

    # Load bytes for every stored document and return as AttachmentContentModel
    from app.repositories.models.conversation import AttachmentContentModel

    result: list[AttachmentContentModel] = []
    for entry in index:
        body = _load_bytes(entry["key"])
        if body is not None:
            result.append(
                AttachmentContentModel(
                    content_type="attachment",
                    body=body,
                    file_name=entry["file_name"],
                )
            )
        else:
            logger.warning(
                "Could not reload persisted document '%s' — skipping",
                entry["file_name"],
            )

    return result
