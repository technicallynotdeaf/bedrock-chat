"""
Hook to limit the number of tool-use cycles in the Strands agent event loop.

Without this, the Strands agent has no built-in max iterations — it will keep
calling tools indefinitely as long as the model requests them. This can cause
runaway token costs and Lambda timeouts.

When the limit is hit, this hook stops the Strands event loop. The stopping
leaves the conversation in a state where the last assistant message contains
only tool_use blocks (no text) — so a safety net in chat_strands.py detects
this and makes a follow-up, tool-free call that instructs the model to
interrogate every tool result and respond to the user.
"""

import logging

from strands.experimental.hooks import AfterToolInvocationEvent
from strands.hooks import HookProvider, HookRegistry

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 10


class MaxTurnsHook(HookProvider):
    """Forces the agent event loop to stop after a maximum number of tool calls.

    The state of this hook can be queried via `limit_reached` so downstream
    code can distinguish "the model naturally finished" from "we cut it off".
    """

    def __init__(self, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS):
        self._max_tool_calls = max_tool_calls
        self._tool_call_count = 0
        self._limit_reached = False

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    @property
    def limit_reached(self) -> bool:
        return self._limit_reached

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolInvocationEvent, self._after_tool)

    def _after_tool(self, event: AfterToolInvocationEvent) -> None:
        self._tool_call_count += 1
        if self._tool_call_count >= self._max_tool_calls:
            self._limit_reached = True
            logger.warning(
                f"Max tool calls reached ({self._tool_call_count}/"
                f"{self._max_tool_calls}). Stopping agent event loop — a "
                "tool-free follow-up call will be made to synthesize a "
                "response from the gathered tool results."
            )
            event.invocation_state.setdefault("request_state", {})
            event.invocation_state["request_state"]["stop_event_loop"] = True
