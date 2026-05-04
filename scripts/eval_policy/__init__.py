"""
LeHome Challenge Policy Module

This module provides the base policy interface and implementations
for the LeHome Challenge evaluation framework.
"""

from .base_policy import BasePolicy
from .registry import PolicyRegistry

# Import policy implementations (this will auto-register them)
from .lerobot_policy import LeRobotPolicy
from .example_participant_policy import CustomPolicy
from .gr00t_policy import GR00TPolicy

__all__ = [
    "BasePolicy",
    "PolicyRegistry",
    "LeRobotPolicy",
    "CustomPolicy",
    "GR00TPolicy",
]
