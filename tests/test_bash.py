"""Tests for the bash session tool runner."""

from shopping.tools.bash import BashSession


class TestBashSession:
    def test_simple_command(self):
        session = BashSession()
        result = session.execute("echo hello")
        assert result == "hello"
        session.close()

    def test_working_directory_persists(self):
        session = BashSession()
        session.execute("cd /tmp")
        result = session.execute("pwd")
        assert result == "/tmp"
        session.close()

    def test_environment_persists(self):
        session = BashSession()
        session.execute("export FOO=bar")
        result = session.execute("echo $FOO")
        assert result == "bar"
        session.close()

    def test_stderr_captured(self):
        session = BashSession()
        result = session.execute("echo oops >&2")
        assert "oops" in result
        session.close()

    def test_multiline_output(self):
        session = BashSession()
        result = session.execute("echo line1; echo line2; echo line3")
        assert result == "line1\nline2\nline3"
        session.close()

    def test_restart(self):
        session = BashSession()
        session.execute("export FOO=bar")
        result = session.execute("echo $FOO")
        assert result == "bar"
        session.restart()
        result = session.execute("echo ${FOO:-empty}")
        assert result == "empty"
        session.close()

    def test_command_with_exit_code(self):
        session = BashSession()
        result = session.execute("ls /nonexistent 2>&1; echo done")
        assert "done" in result
        session.close()

    def test_no_trailing_newline(self):
        """Commands like pbpaste that don't end with a newline."""
        session = BashSession()
        result = session.execute("printf 'no newline here'")
        assert result == "no newline here"
        assert "__SENTINEL" not in result
        session.close()
