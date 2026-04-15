"""
Main chat function for Strands integration.
"""

import json
import logging
from typing import Callable

from app.agents.tools.agent_tool import ToolRunResult
from app.bedrock import calculate_price, BedrockGuardrailsModel
from app.repositories.models.conversation import SimpleMessageModel
from app.repositories.models.custom_bot import (
    BotModel,
    GenerationParamsModel,
)
from app.routes.schemas.conversation import ChatInput
from app.strands_integration.agent import create_strands_agent
from app.strands_integration.converters import (
    simple_message_models_to_strands_messages,
    strands_message_to_simple_message_model,
    strands_message_to_message_model,
)
from app.strands_integration.handlers import MaxTurnsHook, ToolResultCapture, create_callback_handler
from app.stream import OnStopInput, OnThinking
from app.utils import get_current_time
from app.vector_search import (
    SearchResult,
)

from strands import Agent
from strands.telemetry.metrics import EventLoopMetrics
from strands.types.event_loop import StopReason
from strands.types.content import Message
from strands.types.exceptions import MaxTokensReachedException

logger = logging.getLogger(__name__)


_FALLBACK_SUMMARY_INSTRUCTION = (
    "IMPORTANT — tool calls have been paused at the research limit. "
    "No more tools are available in this turn.\n\n"
    "Your task now:\n"
    "1. Carefully read EVERY tool result in the conversation above "
    "(search results, fetched pages, extracted content, crawled pages). "
    "Consider both successful results AND any that failed or errored.\n"
    "2. Using only that gathered content, write a clear, substantive "
    "answer to the user's original question. Include specific facts, "
    "details, and quotes from the results — do not just say that "
    "research was done.\n"
    "3. Cite sources inline with [^source_id] notation where available.\n"
    "4. Briefly note any sources that were unavailable or returned errors, "
    "so the user understands any gaps in the research.\n"
    "5. At the end of your response, ask the user whether they would like "
    "you to continue researching for more detail, or whether the current "
    "answer is sufficient.\n\n"
    "Do not apologise for the research limit — simply answer and offer to "
    "dig deeper."
)


def _has_text_content(message: Message) -> bool:
    """Return True if the message has at least one non-empty text block."""
    for content in message.get("content", []):
        if "text" in content:
            text = content.get("text") or ""
            if text.strip():
                return True
    return False


def _merge_metrics(
    primary: EventLoopMetrics, secondary: EventLoopMetrics
) -> EventLoopMetrics:
    """Merge accumulated usage/metrics from a fallback call into the primary."""
    try:
        for key, value in secondary.accumulated_usage.items():
            primary.accumulated_usage[key] = (
                primary.accumulated_usage.get(key, 0) + value
            )
    except Exception as e:
        logger.warning(f"Could not merge fallback metrics: {e}")
    return primary


def _run_fallback_response(
    agent: Agent,
    on_stream: Callable[[str], None] | None,
    on_reasoning: Callable[[str], None] | None,
    on_message: Callable[[Message], None] | None,
) -> tuple[Message, EventLoopMetrics]:
    """
    Make a follow-up model call WITHOUT tools to force a final text response.

    Used when the main event loop stops without the model producing text
    (e.g. MaxTurnsHook cut it off while it was still calling tools).

    The key constraint: agent.messages ends with a role="user" tool_result
    message, so we must NOT pass any new user message — two consecutive user
    messages would be rejected by Bedrock. Instead, the summary instruction
    is injected via the system prompt and the agent is called with no
    additional input so it responds as the next assistant turn.
    """
    # Combine the original system prompt with the fallback instruction so the
    # model knows what to do without any extra user message.
    base_system = agent.system_prompt or ""
    combined_system = (
        f"{base_system}\n\n{_FALLBACK_SUMMARY_INSTRUCTION}"
        if base_system
        else _FALLBACK_SUMMARY_INSTRUCTION
    )

    # Build a minimal agent with NO tools so the model must produce text.
    fallback_agent = Agent(
        model=agent.model,
        tools=[],
        hooks=[],
        system_prompt=combined_system,
        messages=list(agent.messages),
    )
    # Reuse the streaming callback so tokens stream to the user in real time.
    fallback_agent.callback_handler = create_callback_handler(
        on_stream=on_stream,
        on_reasoning=on_reasoning,
        on_message=on_message,
    )

    # Call with no new prompt — the existing history (ending with the
    # tool_result user message) is used as-is; the model responds as
    # the next assistant turn guided by the system prompt instruction.
    result = fallback_agent()
    return result.message, result.metrics


def converse_with_strands(
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
    Chat with Strands agents.

    Architecture Overview:

    1. Reasoning Content:
       - Streaming: CallbackHandler processes reasoning events for real-time display.
       - Persistence: CallbackHandler notifies the message including reasoning content.

    2. Tool Use/Result (Thinking Log):
       - Streaming: ToolResultCapture processes tool events for real-time display.
       - Persistence: CallbackHandler notifies the message including tool use/result content.

    3. Related Documents (Citations):
       - Source: ToolResultCapture notifies related document.
       - Reason: Requires access to raw tool results for source_link extraction

    Why This Hybrid Approach:

    - ToolResultCapture: Processes raw tool results during execution hooks, enabling
      source_link extraction and citation functionality.

    - CallbackHandler: Captures all messages including reasoning / tool use/result content
      that may not be available in final AgentResult when tools are used.
    """

    tool_capture = ToolResultCapture(
        display_citation=display_citation,
        on_thinking=on_thinking,
        on_tool_result=on_tool_result,
    )
    max_turns_hook = MaxTurnsHook(max_tool_calls=10)

    prompt_caching_enabled = bot.prompt_caching_enabled if bot is not None else True
    has_tools = bot is not None and bot.is_agent_enabled()

    agent = create_strands_agent(
        bot=bot,
        instructions=instructions,
        model_name=chat_input.message.model,
        generation_params=generation_params,
        guardrail=guardrail,
        enable_reasoning=chat_input.enable_reasoning,
        prompt_caching_enabled=prompt_caching_enabled,
        has_tools=has_tools,
        hooks=[tool_capture, max_turns_hook],
    )

    thinking_log: list[SimpleMessageModel] = []

    def on_message(message: Message):
        if any(
            "toolUse" in content or "toolResult" in content
            for content in message["content"]
        ):
            thinking_log.append(strands_message_to_simple_message_model(message))

    agent.callback_handler = create_callback_handler(
        on_stream=on_stream,
        on_reasoning=on_reasoning,
        on_message=on_message,
    )

    # Convert SimpleMessageModel list to Strands Messages format
    strands_messages = simple_message_models_to_strands_messages(
        simple_messages=messages,
        model=chat_input.message.model,
        guardrail=guardrail,
        search_results=search_results,
        prompt_caching_enabled=prompt_caching_enabled,
    )

    def run_agent(agent: Agent) -> tuple[StopReason, Message, EventLoopMetrics]:
        try:
            result = agent(strands_messages)
            return (
                result.stop_reason,
                result.message,
                result.metrics,
            )

        except MaxTokensReachedException:
            return (
                "max_tokens",
                agent.messages[-1],
                agent.event_loop_metrics,
            )

    stop_reason, result_message, metrics = run_agent(agent)

    # Safety net: if the event loop finished without a final text response
    # from the model, make a tool-free follow-up call so the user always
    # sees an answer. This covers:
    #   1. MaxTurnsHook triggered — the model was still calling tools when
    #      we hit the 10-call budget, so Strands returned the last tool_use
    #      message (no text).
    #   2. The model stopped for any other reason without producing text.
    #
    # If the model already produced text alongside tool_use blocks, we keep
    # that response as-is to avoid duplicating work.
    if not _has_text_content(result_message):
        if max_turns_hook.limit_reached:
            logger.warning(
                f"Tool-call limit hit ({max_turns_hook.tool_call_count}) "
                "and no text response was produced. Running tool-free "
                "follow-up so the agent interrogates every tool result "
                "and responds to the user."
            )
        else:
            logger.warning(
                f"Event loop stopped with no text response "
                f"(stop_reason={stop_reason}). "
                "Running tool-free follow-up to produce a final response."
            )

        try:
            fallback_message, fallback_metrics = _run_fallback_response(
                agent=agent,
                on_stream=on_stream,
                on_reasoning=on_reasoning,
                on_message=on_message,
            )
            if _has_text_content(fallback_message):
                result_message = fallback_message
                # Stop reason is now end_turn since the fallback produced text
                stop_reason = "end_turn"
                # Merge metrics so token counts / price reflect both calls
                metrics = _merge_metrics(metrics, fallback_metrics)
                logger.info(
                    "Fallback call produced a text response. "
                    f"Added tokens — input: {fallback_metrics.accumulated_usage.get('inputTokens', 0)}, "
                    f"output: {fallback_metrics.accumulated_usage.get('outputTokens', 0)}."
                )
            else:
                logger.warning(
                    "Fallback call still returned no text. Keeping original "
                    "result message."
                )
        except Exception as e:
            logger.error(
                f"Fallback response call failed: {e}. "
                "Returning original result message.",
                exc_info=True,
            )

    # Convert Strands Message to MessageModel
    message = strands_message_to_message_model(
        message=result_message,
        model_name=chat_input.message.model,
        create_time=get_current_time(),
        thinking_log=thinking_log,
    )

    # Extract token usage from metrics
    input_tokens = metrics.accumulated_usage.get("inputTokens", 0)
    output_tokens = metrics.accumulated_usage.get("outputTokens", 0)
    cache_read_input_tokens = metrics.accumulated_usage.get("cacheReadInputTokens", 0)
    cache_write_input_tokens = metrics.accumulated_usage.get("cacheWriteInputTokens", 0)

    # Calculate price using the same function as chat_legacy
    price = calculate_price(
        model=chat_input.message.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
    )

    logger.info(
        f"token count: {json.dumps({
            'input': input_tokens,
            'output': output_tokens,
            'cache_read_input': cache_read_input_tokens,
            'cache_write_input': cache_write_input_tokens
        })}"
    )
    logger.info(f"price: {price}")

    return OnStopInput(
        message=message,
        stop_reason=stop_reason,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        cache_read_input_count=cache_read_input_tokens,
        cache_write_input_count=cache_write_input_tokens,
        price=price,
    )
