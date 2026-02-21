"""Quick entry point to test the agent loop."""

import asyncio
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from mac.loop import agent_loop
from mac.tool import MacTool


async def main():
    load_dotenv()
    prompt = " ".join(sys.argv[1:]) or "Take a screenshot and describe what you see."
    client = Anthropic()
    tool = MacTool()
    await agent_loop(prompt=prompt, client=client, tool=tool)


if __name__ == "__main__":
    asyncio.run(main())
