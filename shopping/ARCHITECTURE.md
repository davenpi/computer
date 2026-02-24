# Shopping Agent Architecture

## Directory Structure

```
mac/                      # Low-level macOS screen control
  loop.py                 # General agent loop — sends messages, dispatches tools
  tool.py                 # MacTool — pyautogui-based computer use executor

shopping/                 # Shopping agent
  agent.py                # Entry point — configures loop with tools + prompt
  prompt.py               # System prompt, shopping-specific instructions
  tools/
    bash.py               # Persistent bash session runner
    text_editor.py        # Text editor tool runner
  docs/
    shopping-brief.md     # Active punch list (the agent's mission)
    product-philosophy.md # Design principles
    ian-lookbook.md       # Background style reference
```

## Tools

The agent gets three tools:

- **computer** (MacTool) — screen control, clicks, typing, screenshots
- **bash** — shell commands, clipboard via `pbpaste`, file I/O
- **text_editor** — Anthropic's text editor tool for reading/writing files

Tool runners live in `shopping/tools/`. These are reusable and independently
testable — can be pulled into a top-level `tools/` module if a second agent needs them.

## I/O Contract

- **Input:** Agent reads `shopping/docs/shopping-brief.md` for its mission.
  Each item in the buy-next queue is an independent search.
- **Output:** One markdown file per search item in `shopping/results/`.
  Each file contains 3-5 options with product name, price, URL, and reasoning.

## Key Decisions

- Each shopping item is an independent agent run — no cross-item dependencies.
- Anti-preferences accumulate from conversation; purchase tracking is deferred.
- Agent runs on Mac Mini (takes over the screen), not on daily driver.
- `mac/loop.py` needs to be extended to dispatch multiple tool types
  (currently only handles MacTool).
