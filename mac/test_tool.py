import base64
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from tool import MacTool, ToolResult


class TestToolResult:
    def test_defaults_to_none(self):
        result = ToolResult()
        assert result.output is None
        assert result.error is None
        assert result.base64_image is None

    def test_fields(self):
        result = ToolResult(output="hello", error="oops", base64_image="abc123")
        assert result.output == "hello"
        assert result.error == "oops"
        assert result.base64_image == "abc123"

    def test_frozen(self):
        result = ToolResult()
        with pytest.raises(AttributeError):
            result.output = "nope"


class TestScreenshot:
    @pytest.fixture
    def tool(self):
        return MacTool()

    @pytest.mark.asyncio
    async def test_returns_base64_image(self, tool):
        result = await tool.screenshot()
        assert result.error is None
        assert result.base64_image is not None

    @pytest.mark.asyncio
    async def test_decoded_image_is_valid_png(self, tool):
        result = await tool.screenshot()
        image_bytes = base64.b64decode(result.base64_image)
        img = Image.open(BytesIO(image_bytes))
        assert img.format == "PNG"

    @pytest.mark.asyncio
    async def test_scaled_dimensions(self, tool):
        if tool._scaling_target is None:
            pytest.skip("No scaling target for this display")
        result = await tool.screenshot()
        image_bytes = base64.b64decode(result.base64_image)
        img = Image.open(BytesIO(image_bytes))
        assert img.width == tool._scaling_target["width"]
        assert img.height == tool._scaling_target["height"]


class TestKey:
    @pytest.fixture
    def tool(self):
        return MacTool()

    @pytest.mark.asyncio
    async def test_missing_text_returns_error(self, tool):
        result = await tool.key(None)
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_invalid_key_returns_error(self, tool):
        result = await tool.key("notarealkey")
        assert result.error is not None
        assert "unrecognized" in result.error

    @pytest.mark.asyncio
    async def test_single_key(self, tool):
        with patch("tool.pyautogui.press") as mock_press:
            result = await tool.key("Return")
            mock_press.assert_called_once_with("return")
        assert result.error is None

    @pytest.mark.asyncio
    async def test_key_combo(self, tool):
        with patch("tool.pyautogui.hotkey") as mock_hotkey:
            result = await tool.key("super+c")
            mock_hotkey.assert_called_once_with("command", "c")
        assert result.error is None

    @pytest.mark.asyncio
    async def test_maps_x11_keys(self, tool):
        assert tool._map_key("super") == "command"
        assert tool._map_key("Return") == "return"
        assert tool._map_key("alt") == "option"
        assert tool._map_key("BackSpace") == "backspace"
