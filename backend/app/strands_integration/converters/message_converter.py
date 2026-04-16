"""
Message conversion utilities for Strands integration.
"""

import logging
from typing import TypeGuard

from app.bedrock import (
    is_prompt_caching_supported,
    is_unsigned_reasoning_content_supported,
)
from app.repositories.models.conversation import (
    ContentModel,
    MessageModel,
    SimpleMessageModel,
    type_model_name,
)
from app.repositories.models.custom_bot_guardrails import BedrockGuardrailsModel
from app.vector_search import SearchResult

from strands.types.content import (
    ContentBlock,
    GuardContent,
    Message,
    Messages,
    Role,
)

from app.strands_integration.converters.content_converter import (
    content_model_to_strands_content_blocks,
    strands_content_block_to_content_model,
)

logger = logging.getLogger(__name__)


def _is_conversation_role(role: str) -> TypeGuard[Role]:
    return role in ["user", "assistant"]


def _to_guardrails_grounding_source(
    search_results: list[SearchResult],
) -> GuardContent | None:
    """Convert search results to Guardrails Grounding source format."""
    return (
        {
            "text": {
                "text": "\n\n".join(x["content"] for x in search_results),
                "qualifiers": ["grounding_source"],
            }
        }
        if len(search_results) > 0
        else None
    )


_ORPHAN_TOOLUSE_PLACEHOLDER = (
    "(previous tool calls were interrupted — continuing from here)"
)


def _sanitize_orphan_tool_uses(messages: Messages) -> Messages:
    """Drop `toolUse` blocks that have no matching `toolResult` in the next msg.

    Older conversations may contain assistant messages whose `toolUse`
    blocks were orphaned (e.g. by a previous `stop_event_loop` code path
    that has since been removed). Bedrock Converse rejects those with:

        "tool_use ids were found without tool_result blocks immediately after"

    We run this sanitizer just before the request is sent so both legacy
    corrupted history AND any fresh corruption are repaired on the fly.
    """
    sanitized: Messages = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            sanitized.append(msg)
            continue

        tool_use_ids = [
            block["toolUse"]["toolUseId"]
            for block in msg.get("content", [])
            if "toolUse" in block
        ]
        if not tool_use_ids:
            sanitized.append(msg)
            continue

        next_msg = messages[i + 1] if i + 1 < len(messages) else None
        next_tool_result_ids: set[str] = set()
        if next_msg and next_msg.get("role") == "user":
            for block in next_msg.get("content", []):
                if "toolResult" in block:
                    next_tool_result_ids.add(block["toolResult"]["toolUseId"])

        missing = [tid for tid in tool_use_ids if tid not in next_tool_result_ids]
        if not missing:
            sanitized.append(msg)
            continue

        logger.warning(
            "Dropping %d orphan toolUse block(s) from assistant message "
            "(ids=%s) — no matching toolResult in the next message.",
            len(missing),
            missing,
        )
        cleaned_content = [
            block
            for block in msg.get("content", [])
            if not ("toolUse" in block and block["toolUse"]["toolUseId"] in missing)
        ]
        has_text = any(
            "text" in b and (b.get("text") or "").strip() for b in cleaned_content
        )
        if not has_text:
            cleaned_content.append({"text": _ORPHAN_TOOLUSE_PLACEHOLDER})
        sanitized.append({"role": "assistant", "content": cleaned_content})

    return sanitized


def simple_message_models_to_strands_messages(
    simple_messages: list[SimpleMessageModel],
    model: type_model_name,
    guardrail: BedrockGuardrailsModel | None = None,
    search_results: list[SearchResult] | None = None,
    prompt_caching_enabled: bool = True,
) -> Messages:
    """Convert SimpleMessageModel list to Strands Messages format."""

    grounding_source = None
    if search_results and guardrail and guardrail.is_guardrail_enabled:
        grounding_source = _to_guardrails_grounding_source(search_results)

    def process_content(c: ContentModel, role: str) -> list[ContentBlock]:
        # Drop unsigned reasoning blocks for DeepSeek R1 and GPT-OSS models
        if (
            not is_unsigned_reasoning_content_supported(model)
            and c.content_type == "reasoning"
            and not getattr(c, "signature", None)
        ):
            return []

        if c.content_type == "text":
            if (
                role == "user"
                and guardrail
                and guardrail.grounding_threshold > 0
                and grounding_source
            ):
                return [
                    {"guardContent": grounding_source},
                    {
                        "guardContent": {
                            "text": {"text": c.body, "qualifiers": ["query"]}
                        }
                    },
                ]

        return content_model_to_strands_content_blocks(c)

    messages: Messages = [
        {
            "role": message.role,
            "content": [
                block
                for c in message.content
                for block in process_content(c, message.role)
            ],
        }
        for message in simple_messages
        if _is_conversation_role(message.role)
    ]

    # Repair orphan toolUse blocks (no matching toolResult after) so Bedrock
    # Converse accepts the conversation.
    messages = _sanitize_orphan_tool_uses(messages)

    # Add message cache points (same logic as legacy bedrock.py)
    if prompt_caching_enabled and is_prompt_caching_supported(model, target="message"):
        for order, message in enumerate(
            filter(lambda m: m["role"] == "user", reversed(messages))
        ):
            if order >= 2:
                break

            message["content"] = [
                *(message["content"]),
                {
                    "cachePoint": {"type": "default"},
                },
            ]
            logger.debug(f"Added message cache point to user message: {message}")

    return messages


def strands_message_to_simple_message_model(message: Message) -> SimpleMessageModel:
    return SimpleMessageModel(
        role=message["role"],
        content=[
            strands_content_block_to_content_model(content)
            for content in message["content"]
        ],
    )


def strands_message_to_message_model(
    message: Message,
    model_name: type_model_name,
    create_time: float,
    thinking_log: list[SimpleMessageModel] | None,
) -> MessageModel:
    """Convert Strands Message to MessageModel."""

    return MessageModel(
        role=message["role"],
        content=[
            strands_content_block_to_content_model(content)
            for content in message["content"]
        ],
        model=model_name,
        children=[],
        parent=None,  # Will be set later
        create_time=create_time,
        feedback=None,
        used_chunks=None,
        thinking_log=thinking_log,
    )
