"""Prompt builder for the shopping agent.

Reads the shopping brief and constructs task prompts for individual
items on the buy-next queue.
"""

from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"


def load_brief() -> str:
    """Load the shopping brief as a string."""
    return (DOCS_DIR / "shopping-brief.md").read_text()


def build_task_prompt(item_number: int) -> str:
    """Build a task prompt for a specific item on the buy-next queue.

    Parameters
    ----------
    item_number : int
        The item number from the brief (1-7).

    Returns
    -------
    str
        A complete prompt for the agent.
    """
    brief = load_brief()

    return f"""\
You are a personal shopper. Your job is to find good options for a \
specific item from the shopping brief below.

<SHOPPING_BRIEF>
{brief}
</SHOPPING_BRIEF>

<TASK>
Find options for item #{item_number} from the "Buy Next" list.

Instructions:
1. Read the item description, budget, and brand preferences carefully.
2. Open Safari and visit 2-3 of the suggested stores (or similar \
quality retailers).
3. Browse visually — look at the product photos and evaluate whether \
each option fits the brief.
4. Find 2-3 good options that match the requirements.
5. Get the URL by running: osascript -e 'tell application "Safari" \
to get URL of current tab of front window'
6. Write each option to a results file AS YOU FIND IT — do not \
wait until the end. Create the file (e.g., results/dark-wash-jeans.md) \
after your first find, then update it as you find more options. \
Do a final cleanup pass at the end if needed.

Output format for each option:
- Product name
- Price
- URL
- One sentence on why it fits the brief

Also note any options you considered but rejected, and why. This helps \
refine future searches.

Important:
- Stay within the stated budget.
- Respect the anti-preferences (no v-necks, no loud patterns, etc.).
- Check sizing using the profile measurements where possible.
- If a store has a sale or promotion, note it.
- Once you have 3 good options written to the file, stop browsing \
and finalize the results. Do not continue optimizing.
- Only include options you actually visited and verified on-site. \
Never recommend products from memory — every option must have a \
real URL you navigated to.
</TASK>
"""


def build_freeform_prompt(task: str) -> str:
    """Build a prompt for a freeform shopping task.

    Parameters
    ----------
    task : str
        A natural language shopping task (e.g., "find me a
        field watch under $100").

    Returns
    -------
    str
        A complete prompt for the agent.
    """
    brief = load_brief()

    return f"""\
You are a personal shopper. Here is the shopper's profile and \
preferences:

<SHOPPING_BRIEF>
{brief}
</SHOPPING_BRIEF>

<TASK>
{task}

Instructions:
1. Open Safari and browse relevant stores.
2. Find 2-3 good options that match the request.
3. Get the URL by running: osascript -e 'tell application "Safari" \
to get URL of current tab of front window'
4. Write each option to a results file AS YOU FIND IT — do not \
wait until the end. Create the file in results/ after your first \
find, then update it as you find more. Do a final cleanup pass \
at the end if needed.

Output format for each option:
- Product name
- Price
- URL
- One sentence on why it's a good fit

Also note any options you rejected and why.

Important:
- Respect the anti-preferences listed in the brief.
- Use the profile for sizing decisions.
- Once you have 3 good options written to the file, stop browsing \
and finalize the results. Do not continue optimizing.
- Only include options you actually visited and verified on-site. \
Never recommend products from memory — every option must have a \
real URL you navigated to.
</TASK>
"""
