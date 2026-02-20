import base64
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypedDict

import pyautogui
from PIL import Image
from Quartz import CGDisplayBounds, CGMainDisplayID

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


class Resolution(TypedDict):
    width: int
    height: int


MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(Enum):
    COMPUTER = "computer"
    API = "api"


class MacTool:
    def __init__(self):
        bounds = CGDisplayBounds(CGMainDisplayID())
        self.width = int(bounds.size.width)
        self.height = int(bounds.size.height)

        ratio = self.width / self.height
        self._scaling_target = None
        for target in MAX_SCALING_TARGETS.values():
            if abs(target["width"] / target["height"] - ratio) < 0.02:
                if target["width"] < self.width:
                    self._scaling_target = target
                break

    async def __call__(self, action: str, text: str | None = None, *args, **kwargs):

        if action == Action.SCREENSHOT.value:
            return await self.screenshot()
        elif action == Action.KEY.value:
            return await self.key(text)

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
                return ToolResult(error=f"screencapture failed: {result.stderr.decode()}")
            if self._scaling_target:
                img = Image.open(tmp_path)
                img = img.resize(
                    (self._scaling_target["width"], self._scaling_target["height"]),
                    Image.LANCZOS,
                )
                img.save(tmp_path, "PNG")
            return ToolResult(
                base64_image=base64.b64encode(open(tmp_path, "rb").read()).decode()
            )

    
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
        return ToolResult()

    def scale_coordinates(self, x: int, y: int):
        pass
