"""
Hook that caps the number of tool-use cycles in the Strands agent event loop.

Strands does not enforce a max iteration count on its own — it will keep
calling tools indefinitely as long as the model requests them. This can cause
runaway token costs and Lambda timeouts.

Design choice: we MUST NOT short-circuit the event loop via
`request_state["stop_event_loop"] = True`. Doing so returns an AgentResult
whose final message is the assistant's `toolUse` block with no matching
`toolResult` user message — saved to DynamoDB, that corrupts the
conversation history and subsequent turns fail with a Bedrock
ValidationException:

    "tool_use ids were found without tool_result blocks immediately after"

Instead, once the soft limit is hit we swap the selected tool for a small
stub (`_LimitReachedStubTool`) that immediately returns a synthetic
"research limit reached — respond now" tool_result. This preserves
tool_use/tool_result pairing perfectly and lets the model naturally produce
a text response (ending with "shall I keep researching?") on the next
model invocation.
"""

import logging
from typing import Any

from strands.experimental.hooks import BeforeToolInvocationEvent
from strands.hooks import HookProvider, HookRegistry
from strands.types.tools import AgentTool, ToolGenerator, ToolResult, ToolSpec, ToolUse

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 30
DEFAULT_RESERVE_FOR_RESPONSE = 5

_LIMIT_REACHED_MESSAGE = (
    "RESEARCH BUDGET REACHED — no more tool calls are available in this turn.\n\n"
    "You MUST now respond to the user in plain text. Do NOT request any more "
    "tool calls.\n\n"
    "CRITICAL: You must provide a COMPLETE, substantive answer to the user's "
    "original request. The user's question must ALWAYS be answered fully. "
    "Your response must:\n\n"
    "1. Carefully read EVERY tool result already gathered in this conversation "
    "(search results, fetched pages, extracted content, crawled pages, etc.). "
    "Include specific facts, figures, names and quotes — do not just say "
    "\"research was done\".\n"
    "2. Synthesise ALL gathered information into a thorough, well-structured "
    "answer that directly addresses the user's question or request.\n"
    "3. Cite sources inline with [^source_id] notation where possible.\n"
    "4. Briefly note any sources that failed or were unavailable so the user "
    "understands any gaps.\n"
    "5. If the task involved writing or generating content (e.g. a document, "
    "report, analysis), produce the full output — do not summarise or "
    "abbreviate.\n"
    "6. End by asking the user whether they would like you to continue "
    "researching for more detail, or whether the current answer is "
    "sufficient.\n\n"
    "Do not apologise for the limit — just deliver the complete answer and "
    "offer to dig deeper."
)


class _LimitReachedStubTool(AgentTool):
    """Synthetic AgentTool that returns a "stop researching" tool_result.

    Used as a drop-in replacement for any real tool once the tool-call
    budget has been exhausted. Because it still emits a proper `ToolResult`
    for the pending `toolUse`, the conversation stays well-formed (every
    `toolUse` has a matching `toolResult`) and the next model invocation
    can produce a normal text reply.
    """

    def __init__(self, original_name: str) -> None:
        super().__init__()
        self._original_name = original_name

    @property
    def tool_name(self) -> str:
        return self._original_name

    @property
    def tool_spec(self) -> ToolSpec:
        return {
            "name": self._original_name,
            "description": "Research budget exhausted; no further calls permitted.",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }

    @property
    def tool_type(self) -> str:
        return "python"

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> ToolGenerator:
        result: ToolResult = {
            "toolUseId": tool_use["toolUseId"],
            "status": "success",
            "content": [{"text": _LIMIT_REACHED_MESSAGE}],
        }
        yield result


class MaxTurnsHook(HookProvider):
    """Caps tool-use cycles by substituting a stub tool after the limit.

    Two-phase budget:
      - Phase 1 (calls 1‥N-R):  real tools run normally.
      - Phase 2 (calls N-R+1‥N): each tool call is replaced with a stub
        that tells the model to stop researching and deliver a complete
        answer. The R reserved calls give the model room to wind down
        gracefully if it tries to batch-call several tools at once.

    Exposes `limit_reached` / `tool_call_count` so downstream code can log
    when the cap was hit.
    """

    def __init__(
        self,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        reserve_for_response: int = DEFAULT_RESERVE_FOR_RESPONSE,
    ) -> None:
        self._max_tool_calls = max_tool_calls
        self._reserve = reserve_for_response
        self._real_limit = max_tool_calls - reserve_for_response
        self._tool_call_count = 0
        self._limit_reached = False

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    @property
    def limit_reached(self) -> bool:
        return self._limit_reached

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolInvocationEvent, self._before_tool)

    def _before_tool(self, event: BeforeToolInvocationEvent) -> None:
        self._tool_call_count += 1
        if self._tool_call_count > self._real_limit:
            if not self._limit_reached:
                logger.warning(
                    f"Tool-call research budget reached "
                    f"({self._real_limit} of {self._max_tool_calls}). "
                    f"Reserving last {self._reserve} call slots for response "
                    "wind-down — substituting stub tool."
                )
            self._limit_reached = True
            original_name = (
                event.selected_tool.tool_name
                if event.selected_tool is not None
                else event.tool_use.get("name", "unknown")
            )
            event.selected_tool = _LimitReachedStubTool(original_name)
