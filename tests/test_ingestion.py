"""PostgreSQL-backed synchronous ingestion integration tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.config import settings
from app.db.session import engine
from app.main import app
from app.models import Document, DocumentChunk


def _text_pdf(text_value: str) -> bytes:
    escaped = text_value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT\n/F1 12 Tf\n72 720 Td\n({escaped}) Tj\nET\n".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(pdf)


def _blank_pdf() -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def ingestion_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    try:
        connection = engine.connect()
        connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        pytest.skip(f"PostgreSQL integration database unavailable: {type(exc).__name__}")

    transaction = connection.begin_nested() if connection.in_transaction() else connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")

    def override_db():
        yield session

    monkeypatch.setattr(settings, "storage_root", tmp_path / "documents")
    app.dependency_overrides[get_db] = override_db
    try:
        yield session
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


def _upload(
    client: TestClient,
    valid_api_key: str,
    filename: str,
    content: bytes,
    mime_type: str,
):
    return client.post(
        "/documents/upload",
        headers={"X-API-Key": valid_api_key},
        files={"file": (filename, content, mime_type)},
    )


def test_txt_ingestion_persists_document_and_chunks(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    response = _upload(client, valid_api_key, "notes.txt", b"Alpha\n\nBeta", "text/plain")
    assert response.status_code == 201
    body = response.json()
    document = ingestion_db.get(Document, UUID(body["document_id"]))
    chunks = ingestion_db.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document.id)
    ).all()
    assert document is not None
    assert document.status == "INDEXED"
    assert document.parser_version == "txt-v1"
    assert document.embedding_model is None
    assert document.size_bytes == 11
    assert body["chunk_count"] == len(chunks) == 1
    assert chunks[0].chunk_metadata["source_filename"] == "notes.txt"


def test_markdown_heading_metadata_is_preserved(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    response = _upload(
        client,
        valid_api_key,
        "guide.md",
        b"# Introduction\nWelcome.\n\n## Setup\nInstall it.\n",
        "text/markdown",
    )
    chunks = ingestion_db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == UUID(response.json()["document_id"]))
        .order_by(DocumentChunk.chunk_index)
    ).all()
    assert response.status_code == 201
    assert [chunk.section_title for chunk in chunks] == ["Introduction", "Setup"]
    assert chunks[1].chunk_metadata["section_title"] == "Setup"


def test_pdf_ingestion_preserves_page_number(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    response = _upload(
        client, valid_api_key, "sample.pdf", _text_pdf("Extracted PDF text"), "application/pdf"
    )
    chunk = ingestion_db.scalar(
        select(DocumentChunk).where(
            DocumentChunk.document_id == UUID(response.json()["document_id"])
        )
    )
    assert response.status_code == 201
    assert chunk is not None
    assert "Extracted PDF text" in chunk.text_content
    assert chunk.page_number == 1
    assert chunk.chunk_metadata["page_number"] == 1


def test_unsupported_extension_is_rejected(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    response = _upload(client, valid_api_key, "data.csv", b"a,b\n1,2", "text/csv")
    assert response.status_code == 400


def test_oversized_upload_is_rejected(
    client: TestClient,
    valid_api_key: str,
    ingestion_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)
    response = _upload(client, valid_api_key, "large.txt", b"x" * (1024 * 1024 + 1), "text/plain")
    assert response.status_code == 413
    assert not list(settings.storage_root.glob(".upload-*.tmp"))


def test_duplicate_content_is_rejected(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    content = b"Identical document content"
    first = _upload(client, valid_api_key, "first.txt", content, "text/plain")
    second = _upload(client, valid_api_key, "second.txt", content, "text/plain")
    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json() == {
        "detail": "Document with identical content already exists",
        "document_id": first.json()["document_id"],
    }


def test_same_filename_with_different_content_is_allowed(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    first = _upload(client, valid_api_key, "same.txt", b"First version", "text/plain")
    second = _upload(client, valid_api_key, "same.txt", b"Second version", "text/plain")
    assert first.status_code == second.status_code == 201
    assert first.json()["document_id"] != second.json()["document_id"]


def test_path_traversal_filename_is_normalized_and_contained(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    response = _upload(client, valid_api_key, "../../evil.txt", b"Safe content", "text/plain")
    document = ingestion_db.get(Document, UUID(response.json()["document_id"]))
    assert response.status_code == 201
    assert document is not None
    assert document.source_name == "evil.txt"
    assert document.storage_path == f"documents/{document.id}/original.txt"
    assert (settings.storage_root / str(document.id) / "original.txt").is_file()
    assert not (settings.storage_root.parent / "evil.txt").exists()


def test_empty_file_is_rejected(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    response = _upload(client, valid_api_key, "empty.txt", b"", "text/plain")
    assert response.status_code == 400
    assert not any(settings.storage_root.iterdir())


def test_pdf_without_extractable_text_is_rejected(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    response = _upload(client, valid_api_key, "blank.pdf", _blank_pdf(), "application/pdf")
    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"]
    assert not any(settings.storage_root.iterdir())


def test_list_and_detail_return_metadata(
    client: TestClient, valid_api_key: str, ingestion_db: Session
) -> None:
    uploaded = _upload(client, valid_api_key, "listed.txt", b"List me", "text/plain").json()
    headers = {"X-API-Key": valid_api_key}
    listing = client.get("/documents", headers=headers)
    detail = client.get(f"/documents/{uploaded['document_id']}", headers=headers)
    missing = client.get(f"/documents/{uuid4()}", headers=headers)
    assert listing.status_code == 200
    listed_document = next(
        item for item in listing.json() if item["id"] == uploaded["document_id"]
    )
    assert listed_document["chunk_count"] == 1
    assert detail.status_code == 200
    assert detail.json()["parser_version"] == "txt-v1"
    assert missing.status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/documents/upload", {"files": {"file": ("a.txt", b"a", "text/plain")}}),
        ("get", "/documents", {}),
        ("get", f"/documents/{uuid4()}", {}),
    ],
)
def test_document_endpoints_require_authentication(
    client: TestClient, ingestion_db: Session, method: str, path: str, kwargs: dict
) -> None:
    response = getattr(client, method)(path, headers={"X-API-Key": "deliberately-wrong"}, **kwargs)
    assert response.status_code == 401
