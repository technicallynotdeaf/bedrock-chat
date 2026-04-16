import logging
import os
from datetime import timezone
from typing import Callable

from botocore.exceptions import ClientError

from app.agents.tools.agent_tool import ToolRunResult
from app.agents.tools.internet_search import InternetSearchInput, _internet_search
from app.agents.utils import get_tools
from app.bedrock import (
    BedrockGuardrailsModel,
    call_converse_api,
    compose_args_for_converse_api,
    is_tooluse_supported,
)
from app.prompt import build_rag_prompt, get_prompt_to_cite_tool_results
from app.repositories.conversation import (
    RecordNotFoundError,
    find_conversation_by_id,
    store_conversation,
    store_related_documents,
)
from app.repositories.conversation_search import find_conversations_by_query
from app.repositories.custom_bot import alias_exists, store_alias
from app.repositories.models.conversation import (
    AttachmentContentModel,
    ConversationModel,
    ImageContentModel,
    MessageModel,
    ReasoningContentModel,
    RelatedDocumentModel,
    SimpleMessageModel,
    TextContentModel,
    ToolResultContentModel,
    ToolUseContentModel,
)
from app.repositories.models.custom_bot import (
    BotAliasModel,
    BotModel,
    GenerationParamsModel,
)
from app.routes.schemas.conversation import (
    ChatInput,
    ChatOutput,
    Chunk,
    Conversation,
    ConversationSearchResult,
    FeedbackOutput,
    MessageOutput,
    SearchHighlight,
    type_model_name,
)
from app.stream import ConverseApiStreamHandler, OnStopInput, OnThinking
from app.usecases.bot import fetch_bot, modify_bot_last_used_time, modify_bot_stats
from app.usecases.global_config import get_title_model
from app.user import User
from app.base_prompt import BASE_SYSTEM_PROMPT
from app.conversation_document_store import (
    find_historical_attachments,
    get_or_update_document_context,
)
from app.document_chunker import process_attachments_for_context_window
from app.pdf_url_handler import download_pdf, extract_pdf_urls
from app.utils import get_aest_now, get_current_time
from app.web_url_handler import extract_web_urls, fetch_urls_content
from app.vector_search import (
    SearchResult,
    search_related_docs,
    search_result_to_related_document,
)
from typing_extensions import deprecated
from ulid import ULID

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_pdf_urls_in_message(message: MessageModel) -> list[str]:
    """Scan text content in a message for PDF URLs, download them, and add as attachments.

    Returns a list of successfully processed PDF URLs for logging/display purposes.
    """
    # Collect all PDF URLs from text content
    pdf_urls: list[str] = []
    for content in message.content:
        if isinstance(content, TextContentModel):
            urls = extract_pdf_urls(content.body)
            pdf_urls.extend(urls)

    if not pdf_urls:
        return []

    logger.info(f"Found {len(pdf_urls)} PDF URL(s) in message: {pdf_urls}")

    # Download PDFs in parallel
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(len(pdf_urls), 5)) as executor:
        downloads = list(executor.map(download_pdf, pdf_urls))

    processed_urls: list[str] = []
    for url, result in zip(pdf_urls, downloads):
        if result is None:
            logger.warning(f"Skipping PDF URL (download failed): {url}")
            continue

        filename, pdf_bytes = result
        attachment = AttachmentContentModel(
            content_type="attachment",
            body=pdf_bytes,
            file_name=filename,
        )

        # Insert attachment before the text content (same as normal file attachments)
        message.content.insert(0, attachment)
        processed_urls.append(url)
        logger.info(f"Added PDF attachment from URL: {url} as {filename}")

    return processed_urls


def process_web_urls_in_message(message: MessageModel) -> list[tuple[str, str]]:
    """Scan text content for web page URLs, fetch their content, and return as context.

    Returns a list of (url, content) tuples for successfully extracted pages.
    PDF URLs are excluded (handled separately by process_pdf_urls_in_message).
    """
    web_urls: list[str] = []
    for content in message.content:
        if isinstance(content, TextContentModel):
            urls = extract_web_urls(content.body)
            web_urls.extend(urls)

    if not web_urls:
        return []

    logger.info(f"Found {len(web_urls)} web URL(s) in message: {web_urls}")
    return fetch_urls_content(web_urls)


def prepare_conversation(
    user: User,
    chat_input: ChatInput,
) -> tuple[str, ConversationModel, BotModel | None]:
    current_time = get_current_time()
    bot = None

    try:
        # Fetch existing conversation
        conversation = find_conversation_by_id(user.id, chat_input.conversation_id)
        logger.info(f"Found conversation: {conversation}")
        parent_id = chat_input.message.parent_message_id
        if chat_input.message.parent_message_id == "system" and chat_input.bot_id:
            # The case editing first user message and use bot
            parent_id = "instruction"
        elif chat_input.message.parent_message_id is None:
            parent_id = conversation.last_message_id
        if chat_input.bot_id:
            logger.info("Bot id is provided. Fetching bot.")
            owned, bot = fetch_bot(user, chat_input.bot_id)
    except RecordNotFoundError:
        # The case for new conversation. Note that editing first user message is not considered as new conversation.
        logger.info(
            f"No conversation found with id: {chat_input.conversation_id}. Creating new conversation."
        )

        initial_message_map = {
            # Dummy system message, which is used for root node of the message tree.
            "system": MessageModel(
                role="system",
                content=[
                    TextContentModel(
                        content_type="text",
                        body="",
                    )
                ],
                model=chat_input.message.model,
                children=[],
                parent=None,
                create_time=current_time,
                feedback=None,
                used_chunks=None,
                thinking_log=None,
            )
        }
        parent_id = "system"
        if chat_input.bot_id:
            logger.info("Bot id is provided. Fetching bot.")
            parent_id = "instruction"
            # Fetch bot and append instruction
            owned, bot = fetch_bot(user, chat_input.bot_id)
            initial_message_map["instruction"] = MessageModel(
                role="instruction",
                content=[
                    TextContentModel(
                        content_type="text",
                        body=bot.instruction,
                    )
                ],
                model=chat_input.message.model,
                children=[],
                parent="system",
                create_time=current_time,
                feedback=None,
                used_chunks=None,
                thinking_log=None,
            )
            initial_message_map["system"].children.append("instruction")

            if not owned:
                try:
                    # Check alias is already created
                    alias_exists(user.id, chat_input.bot_id)
                except RecordNotFoundError:
                    logger.info(
                        "Bot is not owned by the user. Creating alias to shared bot."
                    )
                    # Create alias item
                    store_alias(user.id, BotAliasModel.from_bot_for_initial_alias(bot))

        # Create new conversation
        conversation = ConversationModel(
            id=chat_input.conversation_id,
            title="New conversation",
            total_price=0.0,
            create_time=current_time,
            message_map=initial_message_map,
            last_message_id="",
            bot_id=chat_input.bot_id,
            should_continue=False,
        )

    # Append user chat input to the conversation
    if not chat_input.continue_generate:
        new_message = MessageModel.from_message_input(chat_input.message)
        new_message.parent = parent_id
        new_message.create_time = current_time

        if chat_input.message.message_id:
            message_id = chat_input.message.message_id
        else:
            message_id = str(ULID())

        conversation.message_map[message_id] = new_message
        conversation.message_map[parent_id].children.append(message_id)  # type: ignore

    # If the "Generate continue" button is pressed, a new_message is not generated.
    else:
        message_id = (
            conversation.message_map[conversation.last_message_id].parent
            or "instruction"
        )

    return (message_id, conversation, bot)


# Maximum number of recent conversation turns (user+assistant pairs) to send
# to the model. Older turns are dropped to reduce input token costs.
# The most recent turn (current user message) is always included.
MAX_HISTORY_TURNS = 20


def trace_to_root(
    node_id: str | None, message_map: dict[str, MessageModel]
) -> list[SimpleMessageModel]:
    """Trace message map from leaf node to root node.

    Only includes thinking_log tool use/results for the most recent 4 messages
    to avoid sending massive tool histories from earlier in the conversation.
    """
    result: list[SimpleMessageModel] = []
    if not node_id or node_id == "system":
        node_id = "instruction" if "instruction" in message_map else "system"

    # First pass: collect all messages
    all_nodes: list[SimpleMessageModel] = []
    thinking_logs: list[list[SimpleMessageModel]] = []
    current_node = message_map.get(node_id)
    while current_node:
        all_nodes.append(SimpleMessageModel.from_message_model(message=current_node))
        logs: list[SimpleMessageModel] = []
        if current_node.thinking_log:
            logs = [
                log
                for log in current_node.thinking_log
                if any(
                    isinstance(content, ToolUseContentModel)
                    or isinstance(content, ToolResultContentModel)
                    for content in log.content
                )
            ]
        thinking_logs.append(logs)

        parent_id = current_node.parent
        if parent_id is None:
            break
        current_node = message_map.get(parent_id)

    # Reverse to get chronological order (root → leaf)
    all_nodes.reverse()
    thinking_logs.reverse()

    # Only include thinking logs for the last 4 messages (2 turns) to save tokens.
    # Older tool use/results are not needed — the model already produced responses
    # based on them.
    for i, (node, logs) in enumerate(zip(all_nodes, thinking_logs)):
        if i >= len(all_nodes) - 4:
            result.extend(logs)
        result.append(node)

    # Apply sliding window: keep only the most recent turns
    if len(result) > 0:
        result = _apply_sliding_window(result, MAX_HISTORY_TURNS)

    return result


def _apply_sliding_window(
    messages: list[SimpleMessageModel],
    max_turns: int,
) -> list[SimpleMessageModel]:
    """Keep only the most recent `max_turns` user/assistant turn pairs.

    Preserves the first message (instruction/system) and always includes
    the current (last) user message.
    """
    if len(messages) <= 2:
        return messages

    # Count user messages (each represents roughly one turn)
    user_count = sum(1 for m in messages if m.role == "user")
    if user_count <= max_turns:
        return messages

    # Keep first message (instruction) + last N turns worth of messages
    first_msg = messages[0] if messages[0].role != "user" else None
    turns_kept = 0
    cut_index = len(messages)

    # Walk backwards counting user messages to find the cut point
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            turns_kept += 1
            if turns_kept >= max_turns:
                cut_index = i
                break

    recent = messages[cut_index:]
    if first_msg is not None and cut_index > 0:
        return [first_msg] + recent
    return recent


def _strip_attachments_from_history(
    messages: list[SimpleMessageModel],
) -> list[SimpleMessageModel]:
    """Remove document/image attachments from all messages except the last user message.

    Bedrock counts total PDF pages across ALL messages in a single API call.
    When conversation history includes previous user messages with PDF attachments,
    the cumulative page count can exceed the model's 100-page limit even though
    each individual document is under 100 pages.

    Since the model already processed those documents in prior turns, stripping
    them from history preserves conversational context (via the text) while
    staying within document limits.
    """
    if not messages:
        return messages

    # Find the index of the last user message (the current input)
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    result = []
    for i, msg in enumerate(messages):
        if i == last_user_idx:
            # Keep the current user message intact
            result.append(msg)
        elif any(
            isinstance(c, (AttachmentContentModel, ImageContentModel))
            for c in msg.content
        ):
            # Strip attachment/image content from historical messages,
            # keeping text and other content types
            filtered_content = [
                c
                for c in msg.content
                if not isinstance(c, (AttachmentContentModel, ImageContentModel))
            ]
            if filtered_content:
                result.append(
                    SimpleMessageModel(role=msg.role, content=filtered_content)
                )
            else:
                # If the message only had attachments, replace with a placeholder
                # to maintain role alternation
                result.append(
                    SimpleMessageModel(
                        role=msg.role,
                        content=[
                            TextContentModel(
                                content_type="text",
                                body="[Document previously provided]",
                            )
                        ],
                    )
                )
        else:
            result.append(msg)

    return result


def _trim_messages_for_context_window(
    messages: list[SimpleMessageModel],
) -> list[SimpleMessageModel]:
    """Remove the oldest user/assistant turn pair to reduce prompt token count.

    Preserves:
    - The first message (system/instruction)
    - The last message (current user input)
    - Valid role alternation after trimming

    Returns a shorter list, or raises ValueError if there is nothing left to trim.
    """
    if len(messages) <= 2:
        raise ValueError(
            "Cannot trim conversation further — only the system prompt and "
            "current message remain, yet the prompt still exceeds the model's "
            "context window. Try using a shorter attachment or starting a new conversation."
        )

    # Find the first removable message (index 1, right after system/instruction).
    # Remove messages from the front of the history (oldest) until we drop at
    # least one complete user+assistant pair so that role alternation stays valid.
    trimmed = list(messages)
    removed = 0
    while len(trimmed) > 2:
        # Remove the message right after the system/instruction message
        trimmed.pop(1)
        removed += 1
        # Check if the message at index 1 (new position) has valid alternation
        # with the system message at index 0.  The first real turn should be a
        # "user" message.  Keep removing until that holds.
        if trimmed[1].role == "user":
            break

    logger.info(
        f"Trimmed {removed} oldest message(s) from conversation history "
        f"({len(messages)} -> {len(trimmed)} messages) to fit context window"
    )
    return trimmed


def _is_prompt_too_long_error(e: Exception) -> bool:
    """Check if the exception is a Bedrock 'prompt is too long' validation error."""
    if isinstance(e, ClientError):
        error_code = e.response.get("Error", {}).get("Code", "")
        error_message = e.response.get("Error", {}).get("Message", "")
        if error_code == "ValidationException" and "prompt is too long" in error_message.lower():
            return True
    # Strands may wrap the error — check the string representation
    error_str = str(e)
    if "prompt is too long" in error_str.lower() and "maximum" in error_str.lower():
        return True
    return False


def chat(
    user: User,
    chat_input: ChatInput,
    on_stream: Callable[[str], None] | None = None,
    on_stop: Callable[[OnStopInput], None] | None = None,
    on_thinking: Callable[[OnThinking], None] | None = None,
    on_tool_result: Callable[[ToolRunResult], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
) -> tuple[ConversationModel, MessageModel]:
    user_msg_id, conversation, bot = prepare_conversation(user, chat_input)
    display_citation = bot is not None and bot.display_retrieved_chunks

    message_map = conversation.message_map

    # Process PDF URLs and web URLs concurrently — they are independent I/O
    web_url_context: list[tuple[str, str]] = []
    if not chat_input.continue_generate:
        user_message = message_map.get(user_msg_id)
        if user_message:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as url_executor:
                pdf_future = url_executor.submit(
                    process_pdf_urls_in_message, user_message
                )
                web_future = url_executor.submit(
                    process_web_urls_in_message, user_message
                )

            processed_pdf_urls = pdf_future.result()
            if processed_pdf_urls:
                logger.info(
                    f"Processed {len(processed_pdf_urls)} PDF URL(s) from user message"
                )
            web_url_context = web_future.result()
            if web_url_context:
                logger.info(
                    f"Extracted content from {len(web_url_context)} web URL(s)"
                )
    instructions: list[str] = (
        [
            content.body
            for content in message_map["instruction"].content
            if isinstance(content, TextContentModel)
        ]
        if "instruction" in message_map
        else []
    )

    # Base system prompt applied to all conversations
    utc_now = get_aest_now().astimezone(timezone.utc)
    instructions.insert(
        0,
        BASE_SYSTEM_PROMPT
        + f"\n\nThe current UTC date and time is {utc_now.strftime('%A, %d %B %Y %H:%M')} UTC. "
        "If the user asks about the current time or date, ask them where they are located or "
        "what timezone they are in, then calculate and provide their local time based on the UTC time above.",
    )

    # Inject extracted web page content into instructions so the model can analyse it
    if web_url_context:
        url_context_lines = [
            "The user's message contains the following web page URL(s). "
            "Their content has been fetched and is provided below for your analysis. "
            "Use this content to answer the user's question about these pages.\n"
        ]
        for url, content in web_url_context:
            url_context_lines.append(
                f"--- Content from {url} ---\n{content}\n--- End of content ---\n"
            )
        instructions.append("\n".join(url_context_lines))

    related_documents: list[RelatedDocumentModel] = []
    search_results: list[SearchResult] = []
    if bot is not None:
        if bot.is_agent_enabled() and is_tooluse_supported(chat_input.message.model):
            if display_citation:
                instructions.append(
                    get_prompt_to_cite_tool_results(
                        model=chat_input.message.model,
                    )
                )
        elif bot.has_knowledge() and not is_tooluse_supported(chat_input.message.model):
            # Fetch most related documents from vector store
            # NOTE: Currently embedding not support multi-modal. For now, use the last content.
            content = conversation.message_map[user_msg_id].content[-1]
            if isinstance(content, TextContentModel):
                pseudo_tool_use_id = "new-message-assistant"

                if on_thinking:
                    on_thinking(
                        {
                            "tool_use_id": pseudo_tool_use_id,
                            "name": "knowledge_base_tool",
                            "input": {
                                "query": content.body,
                            },
                        }
                    )

                search_results = search_related_docs(bot=bot, query=content.body)
                logger.info(f"Search results from vector store: {search_results}")

                if on_tool_result:
                    on_tool_result(
                        {
                            "tool_use_id": pseudo_tool_use_id,
                            "status": "success",
                            "related_documents": [
                                search_result_to_related_document(
                                    search_result=result,
                                    source_id_base=pseudo_tool_use_id,
                                )
                                for result in search_results
                            ],
                        }
                    )

                # Insert contexts to instruction
                instructions.append(
                    build_rag_prompt(
                        search_results=search_results,
                        model=chat_input.message.model,
                        display_citation=display_citation,
                    )
                )

    # Internet search: when enabled, search the web and inject results as context.
    #
    # For normal chat (no bot) on the Strands path, skip this pre-hook — the
    # model is given the ``internet_search`` tool directly so it can invoke
    # searches on any turn with a properly scoped query. The old pre-hook used
    # the raw user text as the query, which produced poor results for
    # follow-up messages like "tell me more about the second one".
    _strands_internet_search_active = (
        bot is None
        and chat_input.enable_internet_search
        and os.environ.get("USE_STRANDS", "true").lower() == "true"
        and is_tooluse_supported(chat_input.message.model)
    )
    if _strands_internet_search_active:
        instructions.append(
            "You have access to an `internet_search` tool for looking up current "
            "information on the web. The user has enabled web search for this "
            "conversation. Use the tool whenever a question would benefit from "
            "up-to-date information or when you are unsure of a factual claim, "
            "including on follow-up turns. Formulate a focused search query that "
            "accounts for the conversation context rather than just the raw "
            "latest message, and cite the sources you use."
        )
    if (
        chat_input.enable_internet_search
        and not chat_input.continue_generate
        and not _strands_internet_search_active
    ):
        user_content = conversation.message_map[user_msg_id].content
        # Use the last text content as the search query
        query_text = next(
            (c.body for c in reversed(user_content) if isinstance(c, TextContentModel)),
            None,
        )
        if query_text:
            logger.info(f"Running internet search for query: {query_text}")
            try:
                search_input = InternetSearchInput(
                    query=query_text,
                    locale="en-us",
                    time_limit="",
                )
                internet_results = _internet_search(
                    tool_input=search_input, bot=bot, model=chat_input.message.model
                )
                if internet_results:
                    # Separate dict results (text summaries) from DocumentToolResultModel (PDFs)
                    from app.repositories.models.conversation import DocumentToolResultModel
                    text_results = [r for r in internet_results if isinstance(r, dict)]
                    pdf_results = [r for r in internet_results if isinstance(r, DocumentToolResultModel)]

                    context_lines = [
                        "The following are recent internet search results relevant to the user's query. "
                        "Use this information to provide an accurate and up-to-date response.\n"
                    ]
                    for i, result in enumerate(text_results, 1):
                        context_lines.append(
                            f"[{i}] {result['source_name']}\n"
                            f"URL: {result['source_link']}\n"
                            f"{result['content']}\n"
                        )
                    instructions.append("\n".join(context_lines))
                    logger.info(
                        f"Injected {len(text_results)} internet search results into instructions"
                    )

                    # Attach any PDFs that were downloaded by the search tool
                    if pdf_results:
                        user_message = message_map.get(user_msg_id)
                        if user_message:
                            for pdf_result in pdf_results:
                                attachment = AttachmentContentModel(
                                    content_type="attachment",
                                    body=pdf_result.document,
                                    file_name=f"{pdf_result.name}.pdf",
                                )
                                user_message.content.insert(0, attachment)
                                logger.info(
                                    f"Attached PDF from search result: {pdf_result.name}"
                                )
            except Exception as e:
                logger.error(f"Internet search failed: {e}. Proceeding without search results.")

    # Leaf node id
    # If `continue_generate` is True, note that new message is not added to the message map.
    node_id = (
        chat_input.message.parent_message_id
        if chat_input.continue_generate
        else message_map[user_msg_id].parent
    )
    if node_id is None:
        raise ValueError("parent_message_id or parent is None")

    messages = trace_to_root(
        node_id=node_id,
        message_map=message_map,
    )

    if chat_input.continue_generate:
        message_for_continue_generate = SimpleMessageModel.from_message_model(
            message=message_map[conversation.last_message_id],
        )

    else:
        messages.append(
            SimpleMessageModel.from_message_model(message=message_map[user_msg_id]),
        )
        message_for_continue_generate = None

    # Re-inject document context from historical messages as a system instruction
    # so the model retains access to uploaded files throughout the conversation.
    # This must run BEFORE _strip_attachments_from_history while the attachment
    # bytes are still present in the message list.
    historical_attachments = find_historical_attachments(messages)
    if historical_attachments:
        doc_context = get_or_update_document_context(
            conversation_id=chat_input.conversation_id,
            historical_attachments=historical_attachments,
        )
        if doc_context:
            instructions.append(doc_context)

    # Strip document/image attachments from historical messages to avoid
    # exceeding Bedrock's 100-page PDF limit across the full conversation
    messages = _strip_attachments_from_history(messages)

    # Chunk large document attachments in the current user message to avoid
    # exceeding the model's context window (~200K tokens)
    if messages:
        last_user_msg = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_msg = messages[i]
                break
        if last_user_msg is not None:
            last_user_msg.content = process_attachments_for_context_window(
                last_user_msg.content
            )

    generation_params = bot.generation_params if bot else None

    # Guardrails
    guardrail = bot.bedrock_guardrails if bot else None

    def on_tool_run_result(run_result: ToolRunResult):
        if run_result["status"] == "success":
            related_documents.extend(run_result["related_documents"])

        if on_tool_result:
            on_tool_result(run_result)

    """
    Routes to Strands or legacy implementation based on USE_STRANDS environment variable.
    Retries with progressively trimmed conversation history if the prompt exceeds the
    model's context window.
    """
    use_strands = os.environ.get("USE_STRANDS", "true").lower() == "true"
    MAX_TRIM_RETRIES = 20

    for attempt in range(MAX_TRIM_RETRIES + 1):
        try:
            if use_strands:
                from app.strands_integration.chat_strands import converse_with_strands

                result = converse_with_strands(
                    bot=bot,
                    chat_input=chat_input,
                    instructions=instructions,
                    generation_params=generation_params,
                    guardrail=guardrail,
                    display_citation=display_citation,
                    messages=messages,
                    search_results=search_results,
                    on_stream=on_stream,
                    on_thinking=on_thinking,
                    on_tool_result=on_tool_run_result,
                    on_reasoning=on_reasoning,
                )
            else:
                result = converse_legacy(
                    bot=bot,
                    chat_input=chat_input,
                    instructions=instructions,
                    generation_params=generation_params,
                    guardrail=guardrail,
                    display_citation=display_citation,
                    messages=messages,
                    search_results=search_results,
                    on_stream=on_stream,
                    on_thinking=on_thinking,
                    on_tool_result=on_tool_run_result,
                    on_reasoning=on_reasoning,
                )
            break  # Success — exit retry loop
        except Exception as e:
            if _is_prompt_too_long_error(e) and attempt < MAX_TRIM_RETRIES:
                logger.warning(
                    f"Prompt too long (attempt {attempt + 1}), "
                    f"trimming oldest messages and retrying..."
                )
                try:
                    messages = _trim_messages_for_context_window(messages)
                except ValueError:
                    # Nothing left to trim — try aggressive chunking on
                    # remaining attachments as a last resort
                    from app.document_chunker import (
                        chunk_attachment,
                        MAX_DOCUMENT_CHARS,
                    )

                    last_user = None
                    for msg in reversed(messages):
                        if msg.role == "user":
                            last_user = msg
                            break

                    if last_user is not None:
                        reduced_limit = MAX_DOCUMENT_CHARS // 2
                        new_content: list = []
                        found_attachment = False
                        for c in last_user.content:
                            if isinstance(c, AttachmentContentModel):
                                found_attachment = True
                                new_content.extend(
                                    chunk_attachment(c, max_total_chars=reduced_limit)
                                )
                            else:
                                new_content.append(c)
                        if found_attachment:
                            last_user.content = new_content
                            logger.info(
                                "Applied aggressive document chunking after "
                                "trim exhaustion"
                            )
                            continue

                        # No attachments found — try truncating large text
                        # blocks (e.g. from prior chunking) as a last resort
                        truncated = False
                        truncated_content: list = []
                        for c in last_user.content:
                            if (
                                isinstance(c, TextContentModel)
                                and len(c.body) > reduced_limit
                            ):
                                truncated = True
                                truncated_content.append(
                                    TextContentModel(
                                        content_type="text",
                                        body=c.body[:reduced_limit]
                                        + "\n\n[Content truncated to fit context window]",
                                    )
                                )
                            else:
                                truncated_content.append(c)
                        if truncated:
                            last_user.content = truncated_content
                            logger.info(
                                "Truncated large text blocks after "
                                "trim exhaustion"
                            )
                            continue

                    # No attachments to chunk — re-raise original error
                    raise e
            else:
                raise

    # Post handling: process the result and update conversation
    return post_process_result(
        result=result,
        message_for_continue_generate=message_for_continue_generate,
        conversation=conversation,
        user_msg_id=user_msg_id,
        bot=bot,
        user=user,
        chat_input=chat_input,
        search_results=search_results,
        related_documents=related_documents,
        on_stop=on_stop,
    )


@deprecated("Use chat() instead")
def converse_legacy(
    bot: BotModel | None,
    chat_input: ChatInput,
    instructions: list[str],
    generation_params: GenerationParamsModel | None,
    guardrail: BedrockGuardrailsModel | None,
    display_citation: bool,
    messages: list[SimpleMessageModel],
    search_results: list[SearchResult],
    on_stream: Callable[[str], None] | None = None,
    on_thinking: Callable[[OnThinking], None] | None = None,
    on_tool_result: Callable[[ToolRunResult], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
) -> OnStopInput:
    """
    Legacy converse implementation.

    WARNING: This implementation is deprecated and will be removed in a future version.
    Please migrate to the Strands-based implementation by setting USE_STRANDS=true.
    """
    tools = get_tools(bot, chat_input.message.model)
    stream_handler = ConverseApiStreamHandler(
        model=chat_input.message.model,
        instructions=instructions,
        generation_params=generation_params,
        guardrail=guardrail,
        tools=tools,
        on_stream=on_stream,
        on_thinking=on_thinking,
        on_reasoning=on_reasoning,
    )

    thinking_log: list[SimpleMessageModel] = []

    continue_generate = chat_input.continue_generate
    input_token_count = 0
    output_token_count = 0
    cache_read_input_count = 0
    cache_write_input_count = 0
    price = 0.0

    while True:
        result: OnStopInput = stream_handler.run(
            messages=messages,
            search_results=search_results,
            enable_reasoning=chat_input.enable_reasoning,
            prompt_caching_enabled=(
                bot.prompt_caching_enabled if bot is not None else True
            ),
        )

        message = result["message"]
        stop_reason = result["stop_reason"]

        input_token_count += result["input_token_count"]
        output_token_count += result["output_token_count"]
        cache_read_input_count += result["cache_read_input_count"]
        cache_write_input_count += result["cache_write_input_count"]
        price += result["price"]

        if stop_reason != "tool_use":  # Tool use converged
            # Retain tool use and its result logs
            tool_logs = [
                log
                for log in thinking_log
                if any(
                    isinstance(content, (ToolUseContentModel, ToolResultContentModel))
                    for content in log.content
                )
            ]
            if tool_logs:
                message.thinking_log = tool_logs

            return OnStopInput(
                message=message,
                stop_reason=stop_reason,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                cache_read_input_count=cache_read_input_count,
                cache_write_input_count=cache_write_input_count,
                price=price,
            )

        tool_use_message = SimpleMessageModel.from_message_model(message=message)
        if continue_generate:
            messages[-1] = tool_use_message

            continue_generate = False

        else:
            messages.append(tool_use_message)

        thinking_log.append(tool_use_message)

        tool_use_contents = [
            content
            for content in tool_use_message.content
            if isinstance(content, ToolUseContentModel)
        ]

        run_results: list[ToolRunResult] = []
        for content in tool_use_contents:
            tool = tools[content.body.name]
            run_result = tool.run(
                tool_use_id=content.body.tool_use_id,
                input=content.body.input,
                model=chat_input.message.model,
                bot=bot,
            )
            run_results.append(run_result)

            if on_tool_result:
                on_tool_result(run_result)

        tool_result_message = SimpleMessageModel(
            role="user",
            content=[
                ToolResultContentModel.from_tool_run_result(
                    run_result=result,
                    model=chat_input.message.model,
                    display_citation=display_citation,
                )
                for result in run_results
            ],
        )
        messages.append(tool_result_message)
        thinking_log.append(tool_result_message)


def post_process_result(
    result: OnStopInput,
    message_for_continue_generate: SimpleMessageModel | None,
    conversation: ConversationModel,
    user_msg_id: str,
    bot: BotModel | None,
    user: User,
    chat_input: ChatInput,
    search_results: list[SearchResult],
    related_documents: list[RelatedDocumentModel],
    on_stop: Callable[[OnStopInput], None] | None = None,
):
    """Post-process OnStopInput and update conversation."""

    message = result["message"]
    stop_reason = result["stop_reason"]

    conversation.total_price += result["price"]
    conversation.should_continue = stop_reason == "max_tokens"

    # Set message parent and generate assistant message ID
    message.parent = user_msg_id

    # Generate assistant message ID
    if chat_input.continue_generate and not message.thinking_log:
        if message_for_continue_generate is not None:
            message.continue_from(message_for_continue_generate)

        assistant_msg_id = conversation.last_message_id
        conversation.message_map[assistant_msg_id] = message

    else:
        if chat_input.continue_generate and message.thinking_log:
            if message_for_continue_generate is not None:
                message.thinking_log[0].continue_from(message_for_continue_generate)

            # Remove old assistant message and create new one
            old_assistant_msg_id = conversation.last_message_id
            conversation.message_map[user_msg_id].children.remove(old_assistant_msg_id)
            del conversation.message_map[old_assistant_msg_id]

        # Issue id for new assistant message
        assistant_msg_id = str(ULID())
        conversation.message_map[assistant_msg_id] = message

        # Append children to parent
        conversation.message_map[user_msg_id].children.append(assistant_msg_id)
        conversation.last_message_id = assistant_msg_id

        # Create related documents with consistent source_id format
        search_results_as_related_documents = [
            search_result_to_related_document(
                search_result=result,
                source_id_base=assistant_msg_id,
            )
            for result in search_results
        ]

        # Store RAG results in ToolResultCapture for citation support
        related_documents.extend(search_results_as_related_documents)

    # Store conversation before finish streaming so that front-end can avoid 404 issue
    store_conversation(user.id, conversation)
    if related_documents:
        store_related_documents(
            user_id=user.id,
            conversation_id=conversation.id,
            related_documents=related_documents,
        )

    # Call on_stop callback
    if on_stop:
        on_stop(result)

    # Update bot statistics
    if bot:
        logger.debug("Bot is provided. Updating bot last used time.")
        # Update bot last used time
        modify_bot_last_used_time(user, bot)
        # Update bot stats
        modify_bot_stats(user, bot, increment=1)

    return conversation, message


def chat_output_from_message(
    conversation: ConversationModel,
    message: MessageModel,
) -> ChatOutput:
    return ChatOutput(
        conversation_id=conversation.id,
        create_time=conversation.create_time,
        message=MessageOutput(
            role=message.role,
            content=[c.to_content() for c in message.content],
            model=message.model,
            children=message.children,
            parent=message.parent,
            feedback=None,
            used_chunks=(
                [
                    Chunk(
                        content=c.content,
                        content_type=c.content_type,
                        source=c.source,
                        rank=c.rank,
                    )
                    for c in message.used_chunks
                ]
                if message.used_chunks
                else None
            ),
            thinking_log=(
                [m.to_schema() for m in message.thinking_log]
                if message.thinking_log
                else None
            ),
        ),
        bot_id=conversation.bot_id,
    )


def propose_conversation_title(
    user_id: str,
    conversation_id: str,
) -> str:
    # Use the configured title model for generating conversation titles
    model = get_title_model()

    PROMPT = """Reading the conversation above, what is the appropriate title for the conversation? When answering the title, please follow the rules below:
<rules>
- Title length must be from 15 to 20 characters.
- Prefer more specific title than general. Your title should always be distinct from others.
- Return the conversation title only. DO NOT include any strings other than the title.
- Title must be in the same language as the conversation.
</rules>
"""
    # Fetch existing conversation
    conversation = find_conversation_by_id(user_id, conversation_id)

    messages = trace_to_root(
        node_id=conversation.last_message_id,
        message_map=conversation.message_map,
    )
    # Strip attachments — title generation doesn't need document content
    messages = _strip_attachments_from_history(messages)

    # Append message to generate title
    new_message = SimpleMessageModel(
        role="user",
        content=[
            TextContentModel(
                content_type="text",
                body=PROMPT,
            )
        ],
    )
    messages.append(new_message)

    # Invoke Bedrock
    args = compose_args_for_converse_api(
        messages=[
            message
            for message in messages
            if not any(
                isinstance(content, ToolUseContentModel)
                or isinstance(content, ToolResultContentModel)
                or isinstance(content, ReasoningContentModel)
                for content in message.content
            )
        ],
        model=model,
        stream=False,
    )
    response = call_converse_api(args)
    reply_txt = (
        response["output"]["message"]["content"][0]["text"]
        if "message" in response["output"]
        and len(response["output"]["message"]["content"]) > 0
        and "text" in response["output"]["message"]["content"][0]
        else ""
    )

    return reply_txt


def fetch_conversation(user_id: str, conversation_id: str) -> Conversation:
    conversation = find_conversation_by_id(user_id, conversation_id)

    message_map = {
        message_id: MessageOutput(
            role=message.role,
            content=[c.to_content() for c in message.content],
            model=message.model,
            children=message.children,
            parent=message.parent,
            feedback=(
                FeedbackOutput(
                    thumbs_up=message.feedback.thumbs_up,
                    category=message.feedback.category,
                    comment=message.feedback.comment,
                )
                if message.feedback
                else None
            ),
            used_chunks=(
                [
                    Chunk(
                        content=c.content,
                        content_type=c.content_type,
                        source=c.source,
                        rank=c.rank,
                    )
                    for c in message.used_chunks
                ]
                if message.used_chunks
                else None
            ),
            thinking_log=(
                [m.to_schema() for m in message.thinking_log]
                if message.thinking_log
                else None
            ),
        )
        for message_id, message in conversation.message_map.items()
    }
    # Omit instruction
    if "instruction" in message_map:
        for c in message_map["instruction"].children:
            message_map[c].parent = "system"
        message_map["system"].children = message_map["instruction"].children

        del message_map["instruction"]

    output = Conversation(
        id=conversation_id,
        title=conversation.title,
        create_time=conversation.create_time,
        last_message_id=conversation.last_message_id,
        message_map=message_map,
        bot_id=conversation.bot_id,
        should_continue=conversation.should_continue,
    )
    return output


def search_conversations(query: str, user: User) -> list[ConversationSearchResult]:
    """Search conversations by keyword"""
    conversations = find_conversations_by_query(query, user)
    output = []

    for conversation in conversations:
        # Convert model SearchHighlightModel to schema SearchHighlight
        schema_highlights = None
        if conversation.highlights:
            schema_highlights = [
                SearchHighlight(
                    field_name=highlight.field_name, fragments=highlight.fragments
                )
                for highlight in conversation.highlights
            ]

        # Create ConversationSearchResult with properly converted highlights
        output.append(
            ConversationSearchResult(
                id=conversation.id,
                title=conversation.title,
                last_updated_time=conversation.last_updated_time,
                bot_id=conversation.bot_id,
                highlights=schema_highlights,
            )
        )

    return output
