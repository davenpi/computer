"""Mac executor for Anthropic's computer use tool.

Maps Anthropic's computer use action space to macOS using pyautogui.
The scaling logic and coordinate mapping are specific to Anthropic's
Vision API and its documented max image dimensions. This is a learning
environment and not intended for production use.
"""

import asyncio
import base64
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Literal

import pyautogui
from PIL import Image
from Quartz import CGDisplayBounds, CGMainDisplayID

logger = logging.getLogger(__name__)

# Map X11 key names (what Claude sends) to pyautogui key names
KEY_MAP = {
    "Return": "return",
    "Tab": "tab",
    "space": "space",
    "BackSpace": "backspace",
    "Delete": "delete",
    "Escape": "escape",
    "super": "command",
    "Super_L": "command",
    "Super_R": "command",
    "ctrl": "ctrl",
    "Control_L": "ctrl",
    "Control_R": "ctrl",
    "alt": "option",
    "Alt_L": "option",
    "Alt_R": "option",
    "shift": "shift",
    "Shift_L": "shift",
    "Shift_R": "shift",
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    "Home": "home",
    "End": "end",
    "Page_Up": "pageup",
    "Page_Down": "pagedown",
}


class ToolError(Exception):
    """Raised when a tool action fails."""

    pass


@dataclass(frozen=True)
class ToolResult:
    """Result of a tool action.

    Attributes
    ----------
    output : str or None
        Text output from the action (e.g. cursor coordinates).
    error : str or None
        Error message if the action failed.
    base64_image : str or None
        Base64-encoded PNG screenshot.
    """

    output: str | None = None
    error: str | None = None
    base64_image: str | None = None

    def with_image(self, base64_image: str | None) -> "ToolResult":
        """Return a new ToolResult with the given screenshot attached."""
        return ToolResult(
            output=self.output, error=self.error, base64_image=base64_image
        )


class Action(Enum):
    KEY = "key"
    TYPE = "type"
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK = "left_click"
    LEFT_CLICK_DRAG = "left_click_drag"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    DOUBLE_CLICK = "double_click"
    SCREENSHOT = "screenshot"
    CURSOR_POSITION = "cursor_position"
    LEFT_MOUSE_DOWN = "left_mouse_down"
    LEFT_MOUSE_UP = "left_mouse_up"
    SCROLL = "scroll"
    HOLD_KEY = "hold_key"
    WAIT = "wait"
    TRIPLE_CLICK = "triple_click"
    ZOOM = "zoom"


ScrollDirection = Literal["up", "down", "left", "right"]


@dataclass(frozen=True)
class ScalingTarget:
    """A target resolution to scale screenshots down to before sending to the API.

    We scale screenshots ourselves rather than letting the API do it,
    so we control the exact mapping between Claude's coordinate space
    and the actual screen coordinates. Targets should stay under the
    API's max dimensions for their aspect ratio.

    Attributes
    ----------
    name : str
        Human-readable name for this target.
    width : int
        Target width in pixels for the scaled screenshot.
    height : int
        Target height in pixels for the scaled screenshot.
    api_max_width : int
        Max width the API accepts at this aspect ratio before resizing.
    api_max_height : int
        Max height the API accepts at this aspect ratio before resizing.
    description : str
        What displays this target is intended for.
    """

    name: str
    width: int
    height: int
    api_max_width: int
    api_max_height: int
    description: str

    @property
    def ratio(self) -> float:
        return self.width / self.height


SCALING_TARGETS = [
    ScalingTarget("1:1", 1024, 1024, 1092, 1092, "Square displays"),
    ScalingTarget("XGA", 1024, 768, 1268, 951, "4:3 displays"),
    ScalingTarget("3:2", 1024, 682, 1344, 896, "3:2 displays"),
    ScalingTarget("MBA13", 1024, 666, 1344, 896, '13" MacBook Air (1470x956)'),
    ScalingTarget("WXGA", 1280, 800, 1344, 896, "16:10 displays"),
    ScalingTarget("FWXGA", 1366, 768, 1456, 819, "~16:9 displays"),
    ScalingTarget("2:1", 1280, 640, 1568, 784, "2:1 ultrawide displays"),
]


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "API"


SCREENSHOT_DELAY = 2.0


class MacTool:
    def __init__(self):
        bounds = CGDisplayBounds(CGMainDisplayID())
        self.width = int(bounds.size.width)
        self.height = int(bounds.size.height)

        ratio = self.width / self.height
        self._scaling_target: ScalingTarget | None = None
        for target in SCALING_TARGETS:
            if abs(target.ratio - ratio) < 0.02:
                if target.width < self.width:
                    self._scaling_target = target
                break

        if self._scaling_target is None:
            logger.warning(
                "No scaling target found for display %dx%d (ratio %.3f). "
                "Screenshots will be sent at full resolution and may be "
                "resized by the API, causing coordinate mapping errors. "
                "Add a ScalingTarget for this display.",
                self.width,
                self.height,
                ratio,
            )

    async def __call__(
        self,
        action: str,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        *args,
        **kwargs,
    ):

        if action == Action.SCREENSHOT.value:
            return await self.screenshot()
        elif action == Action.KEY.value:
            return await self.key(text)
        elif action == Action.MOUSE_MOVE.value:
            return await self.mouse_move(text, coordinate)

    async def screenshot(self) -> ToolResult:
        """Capture the screen and return a scaled, base64-encoded PNG.

        Returns
        -------
        ToolResult
            Contains base64_image on success, error on failure.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tmp_path = tmp.name
            result = subprocess.run(
                ["screencapture", "-x", tmp_path], capture_output=True
            )
            if result.returncode != 0:
                return ToolResult(
                    error=f"screencapture failed: {result.stderr.decode()}"
                )
            if self._scaling_target:
                img = Image.open(tmp_path)
                img = img.resize(
                    (self._scaling_target.width, self._scaling_target.height),
                    Image.LANCZOS,
                )
                img.save(tmp_path, "PNG")
            return ToolResult(
                base64_image=base64.b64encode(open(tmp_path, "rb").read()).decode()
            )

    async def _result_with_screenshot(
        self, result: ToolResult, take_screenshot: bool = True
    ) -> ToolResult:
        """Optionally attach a screenshot to a result after a delay.

        Parameters
        ----------
        result : ToolResult
            The action result to attach a screenshot to.
        take_screenshot : bool
            If True, wait for the screen to settle and attach a screenshot.
        """
        if not take_screenshot:
            return result
        await asyncio.sleep(SCREENSHOT_DELAY)
        screenshot = await self.screenshot()
        return result.with_image(screenshot.base64_image)

    def _map_key(self, key: str) -> str:
        """Map an X11 key name to a pyautogui key name."""
        return KEY_MAP.get(key, key.lower())

    async def key(self, text: str | None = None) -> ToolResult:
        """Press a key or key combination.

        Parameters
        ----------
        text : str
            Key name or combo, e.g. "Return", "super+c", "ctrl+shift+t".

        Returns
        -------
        ToolResult
            Error on failure, empty output on success.
        """
        if text is None:
            return ToolResult(error="text is required for key")
        keys = [self._map_key(k) for k in text.split("+")]
        invalid = [k for k in keys if k not in pyautogui.KEYBOARD_KEYS]
        if invalid:
            return ToolResult(error=f"unrecognized key(s): {', '.join(invalid)}")
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
        return await self._result_with_screenshot(ToolResult())

    async def mouse_move(
        self, text: str | None = None, coordinate: tuple[int, int] | None = None
    ) -> ToolResult:
        """Move the mouse to a specific coordinate."""
        if text is not None:
            return ToolResult(error="Text is not supported for mouse move")
        if coordinate is None:
            return ToolResult(error="Coordinate is required for mouse move")
        try:
            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )
        except ToolError as e:
            return ToolResult(error=str(e))
        pyautogui.moveTo(x, y)
        return await self._result_with_screenshot(ToolResult())

    def scale_coordinates(
        self, source: ScalingSource, x: int, y: int
    ) -> tuple[int, int]:
        """Scale coordinates between API space and screen space.

        Parameters
        ----------
        source : ScalingSource
            Where the coordinates came from.
            API: coordinates from Claude, scale up to screen.
            COMPUTER: coordinates from screen, scale down for Claude.
        x : int
            X coordinate.
        y : int
            Y coordinate.

        Returns
        -------
        tuple[int, int]
            Scaled (x, y) coordinates.

        Notes
        -----
        Linear scaling is correct here even though screenshots are resized
        with LANCZOS. LANCZOS affects pixel color blending, not position
        mapping. A point at 50% width in one space is 50% in the other
        regardless of the resampling filter used.
        """
        if source == ScalingSource.API:
            max_x = self._scaling_target.width if self._scaling_target else self.width
            max_y = self._scaling_target.height if self._scaling_target else self.height
            if x < 0 or y < 0 or x > max_x or y > max_y:
                raise ToolError(
                    f"Coordinates ({x}, {y}) are out of bounds (max {max_x}x{max_y})"
                )

        if self._scaling_target is None:
            return x, y

        x_scale = self._scaling_target.width / self.width
        y_scale = self._scaling_target.height / self.height

        if source == ScalingSource.API:
            return round(x / x_scale), round(y / y_scale)
        else:
            return round(x * x_scale), round(y * y_scale)
