"""Shopping agent: multi-tool agent loop with computer use, bash, and text editor.

Adapted from mac/loop.py to dispatch tool calls to the appropriate runner
based on tool name.
"""

import asyncio
import logging
import platform
import time
from dataclasses import dataclass
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

from .prompt import build_freeform_prompt, build_task_prompt
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

# Opus pricing per token (as of Feb 2026)
COST_PER_INPUT_TOKEN = 5.0 / 1_000_000
COST_PER_OUTPUT_TOKEN = 25.0 / 1_000_000


@dataclass
class UsageTracker:
    """Accumulates token usage and cost across API calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    api_seconds: float = 0.0
    wall_seconds: float = 0.0
    iterations: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def record(self, response: BetaMessage, api_time: float) -> float:
        """Record usage from a single API response.

        Returns
        -------
        float
            Cost of this step in USD.
        """
        usage = response.usage
        cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        step_cost = (
            usage.input_tokens * COST_PER_INPUT_TOKEN
            + usage.output_tokens * COST_PER_OUTPUT_TOKEN
            + cache_created * COST_PER_INPUT_TOKEN * 1.25
            + cache_read * COST_PER_INPUT_TOKEN * 0.1
        )
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_creation_input_tokens += cache_created
        self.cache_read_input_tokens += cache_read
        self.api_calls += 1
        self.api_seconds += api_time
        self.iterations += 1
        return step_cost

    @property
    def cost(self) -> float:
        """Estimated cost in USD."""
        return (
            self.input_tokens * COST_PER_INPUT_TOKEN
            + self.output_tokens * COST_PER_OUTPUT_TOKEN
            + self.cache_creation_input_tokens * COST_PER_INPUT_TOKEN * 1.25
            + self.cache_read_input_tokens * COST_PER_INPUT_TOKEN * 0.1
        )

    def step_summary(
        self, iteration: int, max_iterations: int, step_cost: float, api_time: float
    ) -> str:
        """One-line summary for the current iteration."""
        return (
            f"[step {iteration}/{max_iterations}] "
            f"step=${step_cost:.4f} "
            f"cumulative=${self.cost:.4f} "
            f"api={api_time:.1f}s "
            f"cumulative_api={self.api_seconds:.1f}s"
        )

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cached."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def summary(self) -> str:
        """Human-readable summary of usage."""
        lines = [
            "=== Usage Summary ===",
            f"Iterations:    {self.iterations}",
            f"API calls:     {self.api_calls}",
            f"Input tokens:  {self.total_input_tokens:,} "
            f"(uncached={self.input_tokens:,} "
            f"cache_write={self.cache_creation_input_tokens:,} "
            f"cache_read={self.cache_read_input_tokens:,})",
            f"Output tokens: {self.output_tokens:,}",
            f"API time:      {self.api_seconds:.1f}s",
            f"Wall time:     {self.wall_seconds:.1f}s",
            f"Est. cost:     ${self.cost:.4f}",
        ]
        return "\n".join(lines)


def _build_system_prompt(cwd: Path) -> str:
    return f"""\
<SYSTEM_CAPABILITY>
* You are utilising a macOS machine using {_ARCH} \
architecture with internet access.
* You can interact with the computer using mouse, \
keyboard, and screenshot tools.
* You have a bash tool for running shell commands.
* You have a text editor tool for reading and writing files.
* Your working directory is {cwd}. Use relative paths \
for all file operations.
* The current date is {_DATE}.
* To open applications, use Spotlight: key combo \
"super+space", then type the app name and press Return.
* The default browser is Safari. You can type URLs \
directly into the address bar.
* To get the current Safari URL, run this bash command: \
osascript -e 'tell application "Safari" to get URL of \
current tab of front window'. This is the fastest and \
most reliable way to read a URL.
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

    usage = UsageTracker()
    loop_start = time.monotonic()
    try:
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
                    "cache_control": {"type": "ephemeral"},
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": THINKING_BUDGET,
                    },
                },
            )
            api_time = time.monotonic() - t0
            step_cost = usage.record(response, api_time)

            logger.info(
                "Response: stop_reason=%s, blocks=%d, api=%.1fs",
                response.stop_reason,
                len(response.content),
                api_time,
            )
            logger.info(usage.step_summary(i + 1, max_iterations, step_cost, api_time))

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

                    result = await _dispatch_tool(
                        tool_name, inputs, mac_tool, bash, editor
                    )

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
                logger.info("Agent finished (no tool calls)")
                break

            # Inject iteration budget awareness
            remaining = max_iterations - (i + 1)
            if remaining <= int(max_iterations * 0.2):
                budget_msg = (
                    f"[{remaining} iterations remaining. "
                    f"Write your results now and stop browsing.]"
                )
            else:
                budget_msg = (
                    f"[Iteration {i + 1}/{max_iterations} — ${usage.cost:.2f} spent]"
                )
            tool_results.append({"type": "text", "text": budget_msg})

            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("Hit max iterations (%d)", max_iterations)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user at iteration %d", i + 1)

    finally:
        usage.wall_seconds = time.monotonic() - loop_start
        logger.info(usage.summary())
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--item",
        type=int,
        help="Item number from the shopping brief (1-7)",
    )
    group.add_argument(
        "--task",
        type=str,
        help="Freeform shopping task",
    )
    group.add_argument(
        "--raw",
        type=str,
        help="Raw prompt (no shopping brief)",
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt and exit without running",
    )
    args = parser.parse_args()

    if args.item:
        prompt = build_task_prompt(args.item)
    elif args.task:
        prompt = build_freeform_prompt(args.task)
    else:
        prompt = args.raw

    if args.dry_run:
        print(prompt)
        return

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
            prompt=prompt,
            display=args.display,
            max_iterations=args.max_iterations,
        )
    )


if __name__ == "__main__":
    main()
