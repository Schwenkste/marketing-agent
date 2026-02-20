"""
Keyword Agent Package

Exportiert root_runner als einfachen Einstiegspunkt:
from keyword_agent import root_runner
"""

from .keyword_agent import root_runner, root_agent

__all__ = [
    "root_runner",
    "root_agent",
]