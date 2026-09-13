"""Deterministic recursive chunking tests."""

from app.ingestion.chunkers import RecursiveCharacterChunker
from app.ingestion.loaders.base import ExtractedUnit


def test_recursive_chunker_preserves_metadata_and_offsets() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=18, chunk_overlap=4)
    chunks = chunker.chunk(
        [
            ExtractedUnit(
                text="Alpha beta gamma delta epsilon zeta.",
                page_number=2,
                section_title="Example",
                metadata={"source_filename": "sample.pdf"},
            )
        ]
    )

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunk.text_content) <= 18 for chunk in chunks)
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert all(chunk.metadata["section_title"] == "Example" for chunk in chunks)
    assert all(chunk.start_offset < chunk.end_offset for chunk in chunks)
