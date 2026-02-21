"""Agent loop for computer use on macOS.

Sends messages to the Claude API, executes computer tool calls via MacTool,
and returns results until the model stops issuing tool calls.
"""

import platform
from datetime import datetime

from anthropic import Anthropic
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

from .tool import MacTool, ToolResult

BETA_FLAG = "computer-use-2025-11-24"
MODEL = "claude-opus-4-6"
MAX_TOKENS = 16384
THINKING_BUDGET = 8192
MAX_ITERATIONS = 50

_ARCH = platform.machine()
_DATE = datetime.today().strftime("%A, %B %-d, %Y")

SYSTEM_PROMPT = f"""\
<SYSTEM_CAPABILITY>
* You are utilising a macOS machine using {_ARCH} \
architecture with internet access.
* You can interact with the computer using mouse, \
keyboard, and screenshot tools.
* The current date is {_DATE}.
</SYSTEM_CAPABILITY>
"""


async def agent_loop(
    *,
    prompt: str,
    client: Anthropic,
    tool: MacTool,
    system: str = SYSTEM_PROMPT,
    model: str = MODEL,
    messages: list[BetaMessageParam] | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> list[BetaMessageParam]:
    """Run the agent loop until the model stops issuing tool calls.

    Parameters
    ----------
    prompt : str
        The user's task for the agent.
    client : Anthropic
        Anthropic API client.
    tool : MacTool
        The computer use tool executor.
    system : str
        System prompt.
    model : str
        Model to use.
    messages : list or None
        Existing conversation to continue. If None, starts fresh with prompt.
    max_iterations : int
        Safety limit on loop iterations.

    Returns
    -------
    list[BetaMessageParam]
        The full conversation history.
    """
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    tool_config = {
        "type": "computer_20251124",
        "name": "computer",
        "display_width_px": tool._scaling_target.width
        if tool._scaling_target
        else tool.width,
        "display_height_px": tool._scaling_target.height
        if tool._scaling_target
        else tool.height,
    }

    for _ in range(max_iterations):
        response = client.beta.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": system}],
            messages=messages,
            tools=[tool_config],
            betas=[BETA_FLAG],
            extra_body={
                "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET}
            },
        )

        # Convert response to params and append as assistant message
        assistant_content = _response_to_params(response)
        messages.append({"role": "assistant", "content": assistant_content})

        # Process tool use blocks
        tool_results: list[BetaToolResultBlockParam] = []
        for block in assistant_content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block["text"])
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                print(f"[tool] {block['name']}: {block.get('input', {}).get('action')}")
                result = await tool(**block.get("input", {}))
                tool_results.append(_make_tool_result(result, block["id"]))

        if not tool_results:
            return messages

        messages.append({"role": "user", "content": tool_results})

    print(f"[warning] hit max iterations ({max_iterations})")
    return messages


def _response_to_params(response: BetaMessage) -> list[BetaContentBlockParam]:
    """Convert API response content blocks to params for message history."""
    params: list[BetaContentBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                params.append(BetaTextBlockParam(type="text", text=block.text))
        elif getattr(block, "type", None) == "thinking":
            params.append(
                {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None),
                    "signature": getattr(block, "signature", None),
                }
            )
        else:
            params.append(block.model_dump())
    return params


def _make_tool_result(result: ToolResult, tool_use_id: str) -> BetaToolResultBlockParam:
    """Convert a ToolResult to an API tool result block."""
    content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False

    if result.error:
        is_error = True
        content = result.error
    else:
        if result.output:
            content.append({"type": "text", "text": result.output})
        if result.base64_image:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )

    return {
        "type": "tool_result",
        "content": content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }
