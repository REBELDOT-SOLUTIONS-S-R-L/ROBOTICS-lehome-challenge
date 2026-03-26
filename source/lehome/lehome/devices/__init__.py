import os
from typing import TYPE_CHECKING

from .device_base import DeviceBase

if TYPE_CHECKING:
    from .keyboard import BiKeyboard, Se3Keyboard
    from .lerobot import BiSO101Leader, SO101Leader

__all__ = [
    "DeviceBase",
    "SO101Leader",
    "BiSO101Leader",
    "Se3Keyboard",
    "BiKeyboard",
]


def __getattr__(name: str):
    if name in {"SO101Leader", "BiSO101Leader"}:
        from .lerobot import BiSO101Leader, SO101Leader

        return {
            "SO101Leader": SO101Leader,
            "BiSO101Leader": BiSO101Leader,
        }[name]

    if name in {"Se3Keyboard", "BiKeyboard"}:
        if os.environ.get("LEHOME_DISABLE_KEYBOARD") == "1":
            raise AttributeError(name)

        from .keyboard import BiKeyboard, Se3Keyboard

        return {
            "Se3Keyboard": Se3Keyboard,
            "BiKeyboard": BiKeyboard,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
