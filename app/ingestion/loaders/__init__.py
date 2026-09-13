"""Document loader selection."""

from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.loaders.pdf import PdfLoader
from app.ingestion.loaders.text import TextLoader

LOADERS: dict[str, DocumentLoader] = {
    "txt": TextLoader(),
    "md": MarkdownLoader(),
    "pdf": PdfLoader(),
}

__all__ = ["LOADERS", "DocumentLoader"]
