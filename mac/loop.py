"""Agent loop for computer use on macOS.

Sends messages to the Claude API, executes computer tool calls via MacTool,
and returns results until the model stops issuing tool calls.
"""

import logging
import platform
import time
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

logger = logging.getLogger(__name__)

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
* To open applications, use Spotlight: key combo \
"super+space", then type the app name and press Return.
* The default browser is Safari. You can type URLs \
directly into the address bar.
* Use "super+l" to focus the browser address bar.
* When viewing a web page, take a screenshot to see \
the current state. Scroll down if needed content is \
not visible.
* Every action (click, type, scroll, key, etc.) \
automatically returns a screenshot of the result. \
You do not need to call screenshot after each action.
* When using a browser, enter full screen for best \
visibility.
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
    only_n_most_recent_images: int = 3,
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
        Existing conversation to continue. If None, starts fresh
        with prompt.
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

    logger.info(
        "Starting agent loop: display=%dx%d, model=%s",
        tool_config["display_width_px"],
        tool_config["display_height_px"],
        model,
    )
    logger.info("Prompt: %s", prompt)

    loop_start = time.monotonic()
    for i in range(max_iterations):
        logger.info("--- Iteration %d/%d ---", i + 1, max_iterations)

        if only_n_most_recent_images:
            _prune_images(messages, only_n_most_recent_images)

        t0 = time.monotonic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": system}],
            messages=messages,
            tools=[tool_config],
            betas=[BETA_FLAG],
            extra_body={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": THINKING_BUDGET,
                }
            },
        )
        api_time = time.monotonic() - t0

        logger.info(
            "Response: stop_reason=%s, blocks=%d, api=%.1fs",
            response.stop_reason,
            len(response.content),
            api_time,
        )

        # Convert response to params and append as assistant message
        assistant_content = _response_to_params(response)
        messages.append({"role": "assistant", "content": assistant_content})

        # Process tool use blocks
        tool_results: list[BetaToolResultBlockParam] = []
        for block in assistant_content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                thinking = block.get("thinking", "")
                logger.info("[thinking] %s", thinking)
            elif block_type == "text":
                logger.info("[assistant] %s", block["text"])
            elif block_type == "tool_use":
                inputs = block.get("input", {})
                action = inputs.get("action")
                detail = _format_tool_input(inputs)
                logger.info("[tool] %s(%s)", action, detail)
                t0 = time.monotonic()
                result = await tool(**inputs)
                tool_time = time.monotonic() - t0
                if result.error:
                    logger.warning(
                        "[tool] error (%.1fs): %s",
                        tool_time,
                        result.error,
                    )
                else:
                    has_img = "screenshot" if result.base64_image else ""
                    has_out = result.output or ""
                    logger.info(
                        "[tool] ok (%.1fs)%s%s",
                        tool_time,
                        f" output={has_out}" if has_out else "",
                        f" [{has_img}]" if has_img else "",
                    )
                tool_results.append(_make_tool_result(result, block["id"]))

        if not tool_results:
            elapsed = time.monotonic() - loop_start
            logger.info(
                "Done in %d iterations, %.1fs total.",
                i + 1,
                elapsed,
            )
            return messages

        messages.append({"role": "user", "content": tool_results})

    logger.warning("Hit max iterations (%d)", max_iterations)
    return messages


def _format_tool_input(inputs: dict) -> str:
    """Format tool inputs for logging, omitting the action."""
    parts = []
    for k, v in inputs.items():
        if k == "action":
            continue
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def _response_to_params(
    response: BetaMessage,
) -> list[BetaContentBlockParam]:
    """Convert API response content blocks to params."""
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


def _prune_images(messages: list[BetaMessageParam], images_to_keep: int) -> None:
    """Remove all but the most recent N images from tool results.

    Older screenshots lose value as the screen changes. Pruning them
    keeps the prompt size manageable and API latency low.

    Parameters
    ----------
    messages : list[BetaMessageParam]
        Conversation history, modified in place.
    images_to_keep : int
        Number of most recent images to retain.
    """
    # Collect all tool_result blocks that contain images
    tool_results = [
        item
        for message in messages
        for item in (message["content"] if isinstance(message["content"], list) else [])
        if isinstance(item, dict) and item.get("type") == "tool_result"
    ]

    # Count total images
    total = sum(
        1
        for tr in tool_results
        for content in tr.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    to_remove = total - images_to_keep
    if to_remove <= 0:
        return

    # Walk forward (oldest first) and strip images
    removed = 0
    for tr in tool_results:
        if removed >= to_remove:
            break
        if not isinstance(tr.get("content"), list):
            continue
        new_content = []
        for content in tr["content"]:
            if (
                isinstance(content, dict)
                and content.get("type") == "image"
                and removed < to_remove
            ):
                removed += 1
                continue
            new_content.append(content)
        tr["content"] = new_content
