"""Quick entry point to test the agent loop."""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from mac.loop import agent_loop
from mac.tool import MacTool

LOG_DIR = Path("logs")


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Take a screenshot and describe what you see.",
    )
    parser.add_argument(
        "-d",
        "--display",
        type=int,
        default=None,
        help="Display number (1-indexed)",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{timestamp}.log"

    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(file_handler)

    logging.getLogger(__name__).info("Logging to %s", log_file)

    client = Anthropic()
    tool = MacTool(display=args.display)
    await agent_loop(prompt=args.prompt, client=client, tool=tool)


if __name__ == "__main__":
    asyncio.run(main())
