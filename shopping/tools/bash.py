"""Persistent bash session for the agent's bash tool.

Maintains a long-running bash process so that working directory, environment
variables, and other shell state carry across commands. Uses sentinel-based
output delimiting to know when a command has finished.
"""

import queue
import subprocess
import threading
import uuid


class BashSession:
    """A persistent bash session that executes commands and captures output.

    Parameters
    ----------
    timeout : int
        Default timeout in seconds for command execution.
    """

    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._process: subprocess.Popen | None = None
        self._output_queue: queue.Queue[str] = queue.Queue()
        self._error_queue: queue.Queue[str] = queue.Queue()
        self._start()

    def _start(self) -> None:
        """Start or restart the bash process."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait()

        self._process = subprocess.Popen(
            ["/bin/bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )
        self._output_queue = queue.Queue()
        self._error_queue = queue.Queue()
        self._start_readers()

    def _start_readers(self) -> None:
        """Spawn daemon threads to read stdout and stderr without blocking."""
        for stream, q in [
            (self._process.stdout, self._output_queue),
            (self._process.stderr, self._error_queue),
        ]:
            t = threading.Thread(target=self._reader, args=(stream, q), daemon=True)
            t.start()

    @staticmethod
    def _reader(stream, q: queue.Queue) -> None:
        """Read lines from a stream and put them on a queue."""
        for line in stream:
            q.put(line)

    def _read_queue(self, q: queue.Queue, sentinel: str, timeout: int) -> str:
        """Read from a queue until a sentinel line is found or timeout expires.

        Parameters
        ----------
        q : queue.Queue
            The queue to read from.
        sentinel : str
            The sentinel string that marks the end of output.
        timeout : int
            Maximum seconds to wait for output.

        Returns
        -------
        str
            Collected output lines joined together.
        """
        lines = []
        while True:
            try:
                line = q.get(timeout=timeout)
            except queue.Empty:
                break
            if line.strip() == sentinel:
                break
            lines.append(line)
        return "".join(lines)

    def execute(self, command: str, timeout: int | None = None) -> str:
        """Execute a command and return its combined stdout and stderr.

        Parameters
        ----------
        command : str
            The shell command to run.
        timeout : int or None
            Timeout in seconds. Uses the session default if None.

        Returns
        -------
        str
            Combined output from the command.
        """
        if self._process is None or self._process.poll() is not None:
            self._start()

        timeout = timeout or self._timeout
        sentinel = f"__SENTINEL_{uuid.uuid4().hex}__"

        # Write the command, then echo the sentinel on both stdout and stderr
        # so we know when the command's output is done on both streams.
        # The printf ensures a newline exists before the sentinel even if
        # the command's output doesn't end with one (e.g. pbpaste).
        full_command = (
            f"{command}\n"
            f"printf '\\n'\n"
            f"echo {sentinel}\n"
            f"printf '\\n' >&2\n"
            f"echo {sentinel} >&2\n"
        )
        self._process.stdin.write(full_command)
        self._process.stdin.flush()

        stdout = self._read_queue(self._output_queue, sentinel, timeout)
        stderr = self._read_queue(self._error_queue, sentinel, timeout)

        result = stdout
        if stderr:
            result = result + stderr
        return result.strip()

    def restart(self) -> str:
        """Kill the current session and start a fresh one.

        Returns
        -------
        str
            Confirmation message.
        """
        self._start()
        return "Bash session restarted."

    def close(self) -> None:
        """Terminate the bash process."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._process = None
