import base64
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from mac.tool import MacTool, ScalingSource, ToolError, ToolResult

MOCK_SCREENSHOT = ToolResult(base64_image="fake_base64")


@pytest.fixture
def mock_screenshot_delay():
    """Mock the screenshot delay to 0 for fast tests."""
    with patch("mac.tool.SCREENSHOT_DELAY", 0):
        yield


@pytest.fixture
def mock_screenshot(mock_screenshot_delay):
    """Mock both the delay and screenshot for non-screenshot tests."""
    with patch.object(
        MacTool, "screenshot", new_callable=AsyncMock, return_value=MOCK_SCREENSHOT
    ):
        yield


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

    def test_with_image(self):
        result = ToolResult(output="hello")
        updated = result.with_image("img_data")
        assert updated.output == "hello"
        assert updated.base64_image == "img_data"


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
        with Image.open(BytesIO(image_bytes)) as img:
            assert img.width == tool._scaling_target.width
            assert img.height == tool._scaling_target.height


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
    async def test_single_key(self, tool, mock_screenshot):
        with patch("mac.tool.pyautogui.press") as mock_press:
            result = await tool.key("Return")
            mock_press.assert_called_once_with("return")
        assert result.error is None
        assert result.base64_image is not None

    @pytest.mark.asyncio
    async def test_key_combo(self, tool, mock_screenshot):
        with patch("mac.tool.pyautogui.hotkey") as mock_hotkey:
            result = await tool.key("super+c")
            mock_hotkey.assert_called_once_with("command", "c")
        assert result.error is None
        assert result.base64_image is not None

    @pytest.mark.asyncio
    async def test_maps_x11_keys(self, tool):
        assert tool._map_key("super") == "command"
        assert tool._map_key("Return") == "return"
        assert tool._map_key("alt") == "option"
        assert tool._map_key("BackSpace") == "backspace"


class TestScaleCoordinates:
    @pytest.fixture
    def tool(self):
        return MacTool()

    def test_no_scaling_target_passthrough(self, tool):
        tool._scaling_target = None
        assert tool.scale_coordinates(ScalingSource.API, 500, 300) == (500, 300)
        assert tool.scale_coordinates(ScalingSource.COMPUTER, 500, 300) == (500, 300)

    def test_api_to_screen_scales_up(self, tool):
        if tool._scaling_target is None:
            pytest.skip("No scaling target for this display")
        # A point in the middle of the scaled image should map to
        # roughly the middle of the real screen
        mid_x = tool._scaling_target.width // 2
        mid_y = tool._scaling_target.height // 2
        sx, sy = tool.scale_coordinates(ScalingSource.API, mid_x, mid_y)
        assert abs(sx - tool.width // 2) <= 1
        assert abs(sy - tool.height // 2) <= 1

    def test_screen_to_api_scales_down(self, tool):
        if tool._scaling_target is None:
            pytest.skip("No scaling target for this display")
        mid_x = tool.width // 2
        mid_y = tool.height // 2
        ax, ay = tool.scale_coordinates(ScalingSource.COMPUTER, mid_x, mid_y)
        assert abs(ax - tool._scaling_target.width // 2) <= 1
        assert abs(ay - tool._scaling_target.height // 2) <= 1

    def test_out_of_bounds_raises_error(self, tool):
        with pytest.raises(ToolError):
            tool.scale_coordinates(ScalingSource.API, 99999, 99999)

    def test_negative_coords_raises_error(self, tool):
        with pytest.raises(ToolError):
            tool.scale_coordinates(ScalingSource.API, -1, 100)

    def test_roundtrip(self, tool):
        if tool._scaling_target is None:
            pytest.skip("No scaling target for this display")
        # API -> screen -> API should give back ~the same coordinates
        original = (400, 300)
        screen = tool.scale_coordinates(ScalingSource.API, *original)
        back = tool.scale_coordinates(ScalingSource.COMPUTER, *screen)
        assert abs(back[0] - original[0]) <= 1
        assert abs(back[1] - original[1]) <= 1


class TestMouseMove:
    @pytest.fixture
    def tool(self):
        return MacTool()

    @pytest.mark.asyncio
    async def test_missing_coordinate_returns_error(self, tool):
        result = await tool.mouse_move(coordinate=None)
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_text_not_accepted(self, tool):
        result = await tool.mouse_move(text="hello", coordinate=(100, 100))
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_moves_with_scaled_coordinates(self, tool, mock_screenshot):
        with patch("mac.tool.pyautogui.moveTo") as mock_move:
            result = await tool.mouse_move(coordinate=(100, 100))
            expected = tool.scale_coordinates(ScalingSource.API, 100, 100)
            mock_move.assert_called_once_with(*expected)
        assert result.error is None
        assert result.base64_image is not None
