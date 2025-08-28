from abc import ABC, abstractmethod

class PDF_Parser(ABC):
    """Abstract Base Class prescribing the interface for all pdf parsers."""

    @abstractmethod
    def _process(self, source: str) -> None:
        """ Parse the document """
        pass

    @abstractmethod
    def ExportDict(self) -> dict:
        """Export parsed document to dict."""
        pass

    @abstractmethod
    def ExportMarkDown(self) -> dict:
        """Export parsed document to MarkDown."""
        pass