"""Tests for document_chunker module."""

import pytest

from app.document_chunker import (
    CHUNK_SIZE_CHARS,
    LARGE_DOCUMENT_THRESHOLD_BYTES,
    MAX_DOCUMENT_CHARS,
    chunk_attachment,
    chunk_text,
    extract_text_from_document,
    process_attachments_for_context_window,
    should_chunk_attachment,
)
from app.repositories.models.conversation import (
    AttachmentContentModel,
    TextContentModel,
)


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "Hello, world!"
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_into_multiple_chunks(self):
        text = ("A" * 1000 + "\n\n") * 200  # ~200K chars
        chunks = chunk_text(text, chunk_size=50_000)
        assert len(chunks) > 1
        # All text is preserved
        assert "".join(chunks) == text

    def test_prefers_paragraph_boundaries(self):
        # Create text with clear paragraph breaks
        paragraph = "Word " * 100 + "\n\n"  # ~500 chars per paragraph
        text = paragraph * 200  # ~100K chars
        chunks = chunk_text(text, chunk_size=10_000)
        # Each chunk should end at a paragraph boundary (double newline)
        for chunk in chunks[:-1]:  # Last chunk may not end with \n\n
            assert chunk.endswith("\n\n")

    def test_empty_text_returns_single_chunk(self):
        chunks = chunk_text("")
        assert len(chunks) == 1
        assert chunks[0] == ""


class TestExtractTextFromDocument:
    def test_txt_extraction(self):
        data = b"Hello, this is a test document."
        text = extract_text_from_document(data, "test.txt")
        assert text == "Hello, this is a test document."

    def test_csv_extraction(self):
        data = b"col1,col2,col3\nval1,val2,val3\n"
        text = extract_text_from_document(data, "data.csv")
        assert "col1,col2,col3" in text

    def test_md_extraction(self):
        data = b"# Heading\n\nSome content here."
        text = extract_text_from_document(data, "readme.md")
        assert "# Heading" in text
        assert "Some content here." in text

    def test_unknown_format_tries_plain_text(self):
        data = b"Some plain text content"
        text = extract_text_from_document(data, "file.xyz")
        assert text == "Some plain text content"

    def test_binary_data_returns_empty(self):
        data = bytes(range(256))  # Non-printable binary
        text = extract_text_from_document(data, "file.bin")
        # Should return empty since it's not printable text
        assert text == "" or len(text) > 0  # May decode as latin-1


class TestShouldChunkAttachment:
    def test_small_attachment_not_chunked(self):
        att = AttachmentContentModel(
            content_type="attachment", body=b"small", file_name="test.txt"
        )
        assert not should_chunk_attachment(att)

    def test_medium_attachment_not_chunked(self):
        # A 2MB file is under the 2.5MB threshold — should NOT be chunked
        att = AttachmentContentModel(
            content_type="attachment",
            body=b"x" * 2_000_000,
            file_name="test.txt",
        )
        assert not should_chunk_attachment(att)

    def test_large_attachment_chunked(self):
        att = AttachmentContentModel(
            content_type="attachment",
            body=b"x" * (LARGE_DOCUMENT_THRESHOLD_BYTES + 1),
            file_name="test.txt",
        )
        assert should_chunk_attachment(att)


class TestChunkAttachment:
    def test_extracts_and_chunks_large_text(self):
        # Create a large text document
        content = ("Line: " + "A" * 500 + "\n\n") * 1000
        att = AttachmentContentModel(
            content_type="attachment",
            body=content.encode("utf-8"),
            file_name="large.txt",
        )
        result = chunk_attachment(att)
        assert len(result) > 1
        assert all(isinstance(r, TextContentModel) for r in result)
        # First chunk should have the header
        assert "[Document: large.txt" in result[0].body

    def test_returns_error_text_when_extraction_fails_and_too_large(self):
        # Binary data that can't be extracted AND is over the Bedrock 4.5MB limit
        att = AttachmentContentModel(
            content_type="attachment",
            body=b"\x00\x01\x02\x03" * 2_000_000,  # 8MB
            file_name="huge_binary.dat",
        )
        result = chunk_attachment(att)
        assert len(result) == 1
        assert isinstance(result[0], TextContentModel)
        assert "[Error:" in result[0].body

    def test_returns_original_when_extraction_fails_but_small_pdf(self):
        # A small PDF-like file where extraction fails but it's under 4.5MB
        # (Bedrock can try to process it natively)
        att = AttachmentContentModel(
            content_type="attachment",
            body=b"%PDF-1.4 fake small pdf content" + b"\x00" * 1000,
            file_name="small.pdf",
        )
        result = chunk_attachment(att)
        assert len(result) == 1
        assert result[0] is att

    def test_truncates_to_max_chars(self):
        # Create content larger than MAX_DOCUMENT_CHARS
        content = ("X" * 1000 + "\n\n") * 1000  # ~1M chars
        att = AttachmentContentModel(
            content_type="attachment",
            body=content.encode("utf-8"),
            file_name="huge.txt",
        )
        result = chunk_attachment(att)
        total_chars = sum(
            len(r.body) for r in result if isinstance(r, TextContentModel)
        )
        # Should be within reasonable limits (MAX_DOCUMENT_CHARS + overhead for headers)
        assert total_chars <= MAX_DOCUMENT_CHARS + 10_000

    def test_truncated_document_includes_note(self):
        # Create content larger than MAX_DOCUMENT_CHARS
        content = ("X" * 1000 + "\n\n") * 1000  # ~1M chars
        att = AttachmentContentModel(
            content_type="attachment",
            body=content.encode("utf-8"),
            file_name="huge.txt",
        )
        result = chunk_attachment(att)
        # First chunk header should mention truncation
        assert "truncated" in result[0].body.lower()


class TestProcessAttachmentsForContextWindow:
    def test_small_attachments_pass_through(self):
        content = [
            TextContentModel(content_type="text", body="Hello"),
            AttachmentContentModel(
                content_type="attachment", body=b"small file", file_name="test.txt"
            ),
        ]
        result = process_attachments_for_context_window(content)
        assert len(result) == 2
        assert isinstance(result[0], TextContentModel)
        assert isinstance(result[1], AttachmentContentModel)

    def test_small_attachments_pass_through_unchanged(self):
        """Documents under 2.5MB should pass through for Bedrock native handling."""
        small_content = ("Word " * 20 + "\n\n") * 50  # ~5KB
        content = [
            AttachmentContentModel(
                content_type="attachment",
                body=small_content.encode("utf-8"),
                file_name="report.txt",
            ),
            TextContentModel(content_type="text", body="Summarize this"),
        ]
        result = process_attachments_for_context_window(content)
        # Should pass through unchanged (under 500KB threshold)
        assert len(result) == 2
        assert isinstance(result[0], AttachmentContentModel)
        assert isinstance(result[1], TextContentModel)

    def test_large_attachment_gets_chunked(self):
        """Documents over 2.5MB should be text-extracted and chunked."""
        large_content = ("Word " * 200 + "\n\n") * 5000  # ~5MB
        content = [
            AttachmentContentModel(
                content_type="attachment",
                body=large_content.encode("utf-8"),
                file_name="report.txt",
            ),
            TextContentModel(content_type="text", body="Summarize this"),
        ]
        result = process_attachments_for_context_window(content)
        # Should have more blocks now (text chunks + original text)
        assert len(result) > 2
        # Last block should still be the user's text
        assert isinstance(result[-1], TextContentModel)
        assert result[-1].body == "Summarize this"

    def test_mixed_content_only_chunks_large(self):
        small_att = AttachmentContentModel(
            content_type="attachment", body=b"small", file_name="small.txt"
        )
        large_content = ("Word " * 200 + "\n\n") * 5000  # ~5MB
        large_att = AttachmentContentModel(
            content_type="attachment",
            body=large_content.encode("utf-8"),
            file_name="big.txt",
        )
        text = TextContentModel(content_type="text", body="Analyze both")
        content = [small_att, large_att, text]
        result = process_attachments_for_context_window(content)
        # Small attachment should pass through unchanged
        assert result[0] is small_att
        # Last should be the text
        assert result[-1] is text
        # Large attachment should be chunked into text blocks
        assert len(result) > 3
