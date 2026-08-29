from __future__ import annotations

__all__ = ["PaperAuthor", "PaperSearchResult"]


def __getattr__(name: str):
    if name in __all__:
        from src.utils.data_model import PaperAuthor, PaperSearchResult

        exports = {
            "PaperAuthor": PaperAuthor,
            "PaperSearchResult": PaperSearchResult,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
