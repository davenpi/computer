"""Text editor tool runner for Anthropic's str_replace_based_edit_tool.

Handles view, str_replace, create, and insert commands. Files are read
and written relative to a configurable working directory.
"""

import os
from pathlib import Path


class TextEditor:
    """Executes text editor tool commands from Claude.

    Parameters
    ----------
    working_directory : str or Path or None
        Base directory for resolving relative paths. Defaults to cwd.
    """

    def __init__(self, working_directory: str | Path | None = None):
        self._cwd = Path(working_directory or os.getcwd()).resolve()

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to the working directory.

        Parameters
        ----------
        path : str
            File or directory path from Claude.

        Returns
        -------
        Path
            Resolved absolute path.
        """
        p = Path(path)
        if p.is_absolute():
            # Claude sometimes sends absolute paths assuming a container
            # environment (e.g. /repo/file.txt). Make them relative so
            # they resolve against our working directory.
            try:
                p = p.relative_to(p.anchor)
            except ValueError:
                pass
        return (self._cwd / p).resolve()

    def execute(self, command: str, **params) -> str:
        """Dispatch a text editor command.

        Parameters
        ----------
        command : str
            One of "view", "str_replace", "create", "insert".
        **params
            Command-specific parameters from Claude's tool call.

        Returns
        -------
        str
            Result message to return to Claude.
        """
        if command == "view":
            return self._view(params.get("path", ""), params.get("view_range"))
        elif command == "str_replace":
            return self._str_replace(
                params.get("path", ""),
                params.get("old_str", ""),
                params.get("new_str", ""),
            )
        elif command == "create":
            return self._create(params.get("path", ""), params.get("file_text", ""))
        elif command == "insert":
            return self._insert(
                params.get("path", ""),
                params.get("insert_line", 0),
                params.get("insert_text", ""),
            )
        else:
            return f"Error: Unknown command '{command}'"

    def _view(self, path: str, view_range: list[int] | None = None) -> str:
        """View file contents or list directory contents.

        Parameters
        ----------
        path : str
            File or directory path.
        view_range : list[int] or None
            Optional [start, end] line range (1-indexed, -1 for end of file).

        Returns
        -------
        str
            File contents with line numbers, or directory listing.
        """
        resolved = self._resolve(path)

        if resolved.is_dir():
            entries = sorted(resolved.iterdir())
            return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)

        if not resolved.is_file():
            return f"Error: File not found: {path}"

        lines = resolved.read_text().splitlines(keepends=True)

        if view_range is not None:
            start, end = view_range
            if end == -1:
                end = len(lines)
            # Convert from 1-indexed to 0-indexed
            start = max(start - 1, 0)
            end = min(end, len(lines))
            selected = lines[start:end]
            offset = start
        else:
            selected = lines
            offset = 0

        numbered = []
        for i, line in enumerate(selected):
            line_num = offset + i + 1
            numbered.append(f"{line_num}: {line.rstrip()}")
        return "\n".join(numbered)

    def _str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Replace a unique string in a file.

        Parameters
        ----------
        path : str
            File path.
        old_str : str
            Text to find (must match exactly once).
        new_str : str
            Replacement text.

        Returns
        -------
        str
            Success or error message.
        """
        resolved = self._resolve(path)
        if not resolved.is_file():
            return f"Error: File not found: {path}"

        content = resolved.read_text()
        count = content.count(old_str)

        if count == 0:
            return (
                "Error: No match found for replacement. "
                "Please check your text and try again."
            )
        elif count > 1:
            return (
                f"Error: Found {count} matches for replacement text. "
                "Please provide more context to make a unique match."
            )

        new_content = content.replace(old_str, new_str, 1)
        resolved.write_text(new_content)
        return "Successfully replaced text at exactly one location."

    def _create(self, path: str, file_text: str) -> str:
        """Create a new file with the given content.

        Parameters
        ----------
        path : str
            File path to create.
        file_text : str
            Content to write.

        Returns
        -------
        str
            Success or error message.
        """
        resolved = self._resolve(path)
        if resolved.exists():
            return f"Error: File already exists: {path}"

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(file_text)
        return f"Successfully created file {path}"

    def _insert(self, path: str, insert_line: int, insert_text: str) -> str:
        """Insert text after a specific line number.

        Parameters
        ----------
        path : str
            File path.
        insert_line : int
            Line number after which to insert (0 for beginning of file).
        insert_text : str
            Text to insert.

        Returns
        -------
        str
            Success or error message.
        """
        resolved = self._resolve(path)
        if not resolved.is_file():
            return f"Error: File not found: {path}"

        lines = resolved.read_text().splitlines(keepends=True)

        if insert_line < 0 or insert_line > len(lines):
            return (
                f"Error: insert_line {insert_line} is out of range. "
                f"Use 0 to insert at the beginning, or 1-{len(lines)} "
                f"to insert after that line."
            )

        # Ensure inserted text ends with newline
        if insert_text and not insert_text.endswith("\n"):
            insert_text += "\n"

        insert_lines = insert_text.splitlines(keepends=True)
        new_lines = lines[:insert_line] + insert_lines + lines[insert_line:]
        resolved.write_text("".join(new_lines))
        return f"Successfully inserted text after line {insert_line}."
