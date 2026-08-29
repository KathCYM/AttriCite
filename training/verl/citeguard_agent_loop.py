"""CiteGuard agent loop matching src.main's legacy JSON action protocol."""

from __future__ import annotations

import json
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser


def _last_json_object(text: str) -> dict[str, Any] | None:
    """Return the last complete outer ``{reason, action}`` object."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("action"), dict)
            and value["action"].get("name")
        ):
            objects.append(value)
    return objects[-1] if objects else None


class CiteGuardJsonActionParser(ToolParser):
    """Adapt src.main's ``{reason, action}`` JSON to the native tool executor."""

    async def extract_tool_calls(self, responses_ids, tools=None):
        del tools
        text = self.tokenizer.decode(responses_ids, skip_special_tokens=True)
        payload = _last_json_object(text)
        action = payload.get("action") if payload else None
        if not isinstance(action, dict) or not action.get("name"):
            return text, []

        arguments = {"action": action["name"]}
        arguments.update({key: value for key, value in action.items() if key != "name"})
        function_call = FunctionCall(
            name="citeguard",
            arguments=json.dumps(arguments, ensure_ascii=False),
        )
        return text, [function_call]


class CiteGuardAgentLoop(ToolAgentLoop):
    """Execute legacy JSON actions and mirror CiteGuard inference termination."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tool_parser = CiteGuardJsonActionParser(self.tokenizer)

    async def _handle_pending_state(
        self,
        agent_data: Any,
        sampling_params: dict[str, Any],
    ) -> AgentState:
        # src.main puts format instructions in the system prompt rather than
        # exposing an OpenAI/Hermes function schema to the model.
        agent_data._active_tool_schemas = []
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _handle_processing_tools_state(self, agent_data: Any) -> AgentState:
        next_state = await super()._handle_processing_tools_state(agent_data)
        state = agent_data.extra_fields.get("citeguard_state", {})
        # Enforce the action budget after execution. veRL's generic assistant
        # turn limit checks too early and otherwise leaves action five unexecuted.
        if state.get("selected", False) or state.get("actions", 0) >= 5:
            return AgentState.TERMINATED
        return next_state
