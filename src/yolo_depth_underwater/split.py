"""Dataset split selector; values match the keys of an Ultralytics data YAML."""

from enum import StrEnum
from typing import Any

class Split(StrEnum):
    """Dataset split selector; values match the keys of an Ultralytics data YAML."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def coerce(cls, split: Any) -> "Split":
        """Return ``split`` as a member, accepting the matching string too."""
        return split if isinstance(split, cls) else cls(split)
