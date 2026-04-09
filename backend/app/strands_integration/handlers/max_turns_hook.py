"""
Hook to limit the number of tool-use cycles in the Strands agent event loop.

Without this, the Strands agent has no built-in max iterations — it will keep
calling tools indefinitely as long as the model requests them. This can cause
runaway token costs and Lambda timeouts.
"""

import logging

from strands.experimental.hooks import AfterToolInvocationEvent
from strands.hooks import HookProvider, HookRegistry

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 10


class MaxTurnsHook(HookProvider):
    """Forces the agent event loop to stop after a maximum number of tool calls."""

    def __init__(self, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS):
        self._max_tool_calls = max_tool_calls
        self._tool_call_count = 0

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolInvocationEvent, self._after_tool)

    def _after_tool(self, event: AfterToolInvocationEvent) -> None:
        self._tool_call_count += 1
        if self._tool_call_count >= self._max_tool_calls:
            logger.warning(
                f"Max tool calls reached ({self._max_tool_calls}). "
                "Stopping agent event loop to prevent runaway costs."
            )
            event.invocation_state.setdefault("request_state", {})
            event.invocation_state["request_state"]["stop_event_loop"] = True
