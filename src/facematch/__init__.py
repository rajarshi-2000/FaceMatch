"""FaceMatch — find photos of specific people inside an image repository."""

from .engine import FaceMatcher, MatchResult
from .sources import resolve_repository

__version__ = "0.1.0"
__all__ = ["FaceMatcher", "MatchResult", "resolve_repository"]
