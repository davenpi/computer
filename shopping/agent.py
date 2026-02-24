"""Shopping agent: multi-tool agent loop with computer use, bash, and text editor.

Adapted from mac/loop.py to dispatch tool calls to the appropriate runner
based on tool name.
"""

import asyncio
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

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
from dotenv import load_dotenv

from mac.tool import MacTool, ToolResult

from .tools.bash import BashSession
from .tools.text_editor import TextEditor

logger = logging.getLogger(__name__)

BETA_FLAG = "computer-use-2025-11-24"
MODEL = "claude-opus-4-6"
MAX_TOKENS = 16384
THINKING_BUDGET = 8192
MAX_ITERATIONS = 50

_ARCH = platform.machine()
_DATE = datetime.today().strftime("%A, %B %-d, %Y")
_CWD = Path.cwd()


def _build_system_prompt(cwd: Path) -> str:
    return f"""\
<SYSTEM_CAPABILITY>
* You are utilising a macOS machine using {_ARCH} \
architecture with internet access.
* You can interact with the computer using mouse, \
keyboard, and screenshot tools.
* You have a bash tool for running shell commands. \
Use pbpaste to read the clipboard after copying text.
* You have a text editor tool for reading and writing files.
* Your working directory is {cwd}. Use relative paths \
for all file operations.
* The current date is {_DATE}.
* To open applications, use Spotlight: key combo \
"super+space", then type the app name and press Return.
* The default browser is Safari. You can type URLs \
directly into the address bar.
* Use "super+l" to focus the browser address bar.
* To copy a URL: focus the address bar with "super+l", \
then copy with "super+c", then use bash with pbpaste \
to read the clipboard.
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


async def run(
    *,
    prompt: str,
    display: int | None = None,
    model: str = MODEL,
    max_iterations: int = MAX_ITERATIONS,
    only_n_most_recent_images: int = 3,
) -> list[BetaMessageParam]:
    """Run the shopping agent with all tools.

    Parameters
    ----------
    prompt : str
        The task for the agent.
    display : int or None
        Display number for MacTool (1-indexed). None for main display.
    model : str
        Model to use.
    max_iterations : int
        Safety limit on loop iterations.
    only_n_most_recent_images : int
        Keep only the N most recent screenshots to manage context.

    Returns
    -------
    list[BetaMessageParam]
        The full conversation history.
    """
    load_dotenv()

    cwd = Path.cwd()
    client = Anthropic()
    mac_tool = MacTool(display=display)
    bash = BashSession()
    editor = TextEditor(working_directory=cwd)
    system = _build_system_prompt(cwd)

    # Tool configs for the API
    computer_config = {
        "type": "computer_20251124",
        "name": "computer",
        "display_width_px": mac_tool._scaling_target.width
        if mac_tool._scaling_target
        else mac_tool.width,
        "display_height_px": mac_tool._scaling_target.height
        if mac_tool._scaling_target
        else mac_tool.height,
    }
    bash_config = {"type": "bash_20250124", "name": "bash"}
    editor_config = {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
    }

    tools = [computer_config, bash_config, editor_config]
    messages: list[BetaMessageParam] = [{"role": "user", "content": prompt}]

    logger.info(
        "Starting agent: display=%dx%d, model=%s, cwd=%s",
        computer_config["display_width_px"],
        computer_config["display_height_px"],
        model,
        cwd,
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
            tools=tools,
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

        assistant_content = _response_to_params(response)
        messages.append({"role": "assistant", "content": assistant_content})

        # Process tool use blocks
        tool_results: list[BetaToolResultBlockParam] = []
        for block in assistant_content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "thinking":
                logger.info("[thinking] %s", block.get("thinking", ""))
            elif block_type == "text":
                logger.info("[assistant] %s", block["text"])
            elif block_type == "tool_use":
                tool_name = block.get("name")
                inputs = block.get("input", {})
                t0 = time.monotonic()

                result = await _dispatch_tool(tool_name, inputs, mac_tool, bash, editor)

                tool_time = time.monotonic() - t0
                if result.error:
                    logger.warning(
                        "[%s] error (%.1fs): %s",
                        tool_name,
                        tool_time,
                        result.error,
                    )
                else:
                    logger.info(
                        "[%s] ok (%.1fs) output=%s%s",
                        tool_name,
                        tool_time,
                        (result.output or "")[:200],
                        " [screenshot]" if result.base64_image else "",
                    )

                tool_results.append(_make_tool_result(result, block["id"]))

        if not tool_results:
            elapsed = time.monotonic() - loop_start
            logger.info(
                "Done in %d iterations, %.1fs total.",
                i + 1,
                elapsed,
            )
            bash.close()
            return messages

        messages.append({"role": "user", "content": tool_results})

    logger.warning("Hit max iterations (%d)", max_iterations)
    bash.close()
    return messages


async def _dispatch_tool(
    tool_name: str,
    inputs: dict,
    mac_tool: MacTool,
    bash: BashSession,
    editor: TextEditor,
) -> ToolResult:
    """Route a tool call to the appropriate runner.

    Parameters
    ----------
    tool_name : str
        The tool name from the API response.
    inputs : dict
        The tool call inputs.
    mac_tool : MacTool
        Computer use tool.
    bash : BashSession
        Bash session tool.
    editor : TextEditor
        Text editor tool.

    Returns
    -------
    ToolResult
        Unified result object.
    """
    if tool_name == "computer":
        action = inputs.get("action", "")
        detail = _format_inputs(inputs, skip=("action",))
        logger.info("[computer] %s(%s)", action, detail)
        return await mac_tool(**inputs)

    elif tool_name == "bash":
        command = inputs.get("command", "")
        restart = inputs.get("restart", False)
        logger.info("[bash] %s", "restart" if restart else command)
        if restart:
            output = bash.restart()
        else:
            output = bash.execute(command)
        return ToolResult(output=output)

    elif tool_name == "str_replace_based_edit_tool":
        command = inputs.get("command", "")
        path = inputs.get("path", "")
        logger.info("[editor] %s %s", command, path)
        params = {k: v for k, v in inputs.items() if k != "command"}
        output = editor.execute(command, **params)
        return ToolResult(output=output)

    else:
        return ToolResult(error=f"Unknown tool: {tool_name}")


def _format_inputs(inputs: dict, skip: tuple = ()) -> str:
    """Format tool inputs for logging."""
    parts = []
    for k, v in inputs.items():
        if k in skip:
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
    """Remove all but the most recent N images from tool results."""
    tool_results = [
        item
        for message in messages
        for item in (message["content"] if isinstance(message["content"], list) else [])
        if isinstance(item, dict) and item.get("type") == "tool_result"
    ]

    total = sum(
        1
        for tr in tool_results
        for content in tr.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    to_remove = total - images_to_keep
    if to_remove <= 0:
        return

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


# --- CLI entry point ---

LOG_DIR = Path("logs")


def main():
    """Run the shopping agent from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Shopping agent with computer use + bash + editor"
    )
    parser.add_argument("prompt", help="Task for the agent")
    parser.add_argument(
        "-d",
        "--display",
        type=int,
        default=None,
        help="Display number (1-indexed)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Max loop iterations",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"shopping_{timestamp}.log"

    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(file_handler)

    logger.info("Logging to %s", log_file)

    asyncio.run(
        run(
            prompt=args.prompt,
            display=args.display,
            max_iterations=args.max_iterations,
        )
    )


if __name__ == "__main__":
    main()
