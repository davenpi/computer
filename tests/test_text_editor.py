"""Tests for the text editor tool runner."""

import pytest

from shopping.tools.text_editor import TextEditor


@pytest.fixture
def tmp_editor(tmp_path):
    """Create a TextEditor rooted in a temporary directory."""
    return TextEditor(working_directory=tmp_path)


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample Python file for testing."""
    content = "def hello():\n    print('hello')\n    return True\n"
    f = tmp_path / "sample.py"
    f.write_text(content)
    return f


class TestView:
    def test_view_file(self, tmp_editor, sample_file):
        result = tmp_editor.execute("view", path=sample_file.name)
        assert "1: def hello():" in result
        assert "2:     print('hello')" in result
        assert "3:     return True" in result

    def test_view_range(self, tmp_editor, sample_file):
        result = tmp_editor.execute("view", path=sample_file.name, view_range=[2, 3])
        assert "1:" not in result
        assert "2:     print('hello')" in result
        assert "3:     return True" in result

    def test_view_range_end_negative_one(self, tmp_editor, sample_file):
        result = tmp_editor.execute("view", path=sample_file.name, view_range=[2, -1])
        assert "1:" not in result
        assert "2:" in result
        assert "3:" in result

    def test_view_directory(self, tmp_editor, tmp_path):
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        (tmp_path / "subdir").mkdir()
        result = tmp_editor.execute("view", path=".")
        assert "a.py" in result
        assert "b.py" in result
        assert "subdir/" in result

    def test_view_missing_file(self, tmp_editor):
        result = tmp_editor.execute("view", path="nope.py")
        assert "Error" in result


class TestStrReplace:
    def test_replace_unique(self, tmp_editor, sample_file):
        result = tmp_editor.execute(
            "str_replace",
            path=sample_file.name,
            old_str="print('hello')",
            new_str="print('world')",
        )
        assert "Successfully" in result
        assert "print('world')" in sample_file.read_text()

    def test_replace_no_match(self, tmp_editor, sample_file):
        result = tmp_editor.execute(
            "str_replace",
            path=sample_file.name,
            old_str="nonexistent text",
            new_str="replacement",
        )
        assert "No match found" in result

    def test_replace_multiple_matches(self, tmp_editor, tmp_path):
        f = tmp_path / "dups.py"
        f.write_text("foo\nfoo\n")
        result = tmp_editor.execute(
            "str_replace", path="dups.py", old_str="foo", new_str="bar"
        )
        assert "Found 2 matches" in result
        # File should be unchanged
        assert f.read_text() == "foo\nfoo\n"

    def test_replace_missing_file(self, tmp_editor):
        result = tmp_editor.execute(
            "str_replace", path="nope.py", old_str="a", new_str="b"
        )
        assert "Error" in result


class TestCreate:
    def test_create_file(self, tmp_editor, tmp_path):
        result = tmp_editor.execute("create", path="new.py", file_text="print('hi')\n")
        assert "Successfully" in result
        assert (tmp_path / "new.py").read_text() == "print('hi')\n"

    def test_create_nested(self, tmp_editor, tmp_path):
        result = tmp_editor.execute(
            "create", path="sub/dir/new.py", file_text="x = 1\n"
        )
        assert "Successfully" in result
        assert (tmp_path / "sub" / "dir" / "new.py").read_text() == "x = 1\n"

    def test_create_existing_file_errors(self, tmp_editor, sample_file):
        result = tmp_editor.execute(
            "create", path=sample_file.name, file_text="overwrite"
        )
        assert "Error" in result


class TestInsert:
    def test_insert_at_beginning(self, tmp_editor, sample_file):
        result = tmp_editor.execute(
            "insert", path=sample_file.name, insert_line=0, insert_text="# header\n"
        )
        assert "Successfully" in result
        content = sample_file.read_text()
        assert content.startswith("# header\n")

    def test_insert_in_middle(self, tmp_editor, sample_file):
        result = tmp_editor.execute(
            "insert",
            path=sample_file.name,
            insert_line=1,
            insert_text="    # comment\n",
        )
        assert "Successfully" in result
        lines = sample_file.read_text().splitlines()
        assert lines[1] == "    # comment"

    def test_insert_out_of_range(self, tmp_editor, sample_file):
        result = tmp_editor.execute(
            "insert", path=sample_file.name, insert_line=999, insert_text="nope"
        )
        assert "Error" in result

    def test_insert_missing_file(self, tmp_editor):
        result = tmp_editor.execute(
            "insert", path="nope.py", insert_line=0, insert_text="x"
        )
        assert "Error" in result
