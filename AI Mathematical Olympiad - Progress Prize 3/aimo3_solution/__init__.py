"""
AIMO3 Solution Package
Complete solution for AI Mathematical Olympiad - Progress Prize 3
"""

__version__ = "1.0.0"
__author__ = "AIMO3 Team"

from .config import Config
from .math_solver import MathSolver
from .latex_parser import LaTeXParser
from .answer_extractor import AnswerExtractor

__all__ = [
    "Config",
    "MathSolver",
    "LaTeXParser",
    "AnswerExtractor",
]
