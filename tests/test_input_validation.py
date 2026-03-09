"""Tests for input validation and bounds checking.

This tests CRITICAL #2: Input Validation Missing
- Empty/whitespace-only strings
- String length limits
- Confidence bounds (0.0-1.0)
- Tag validation
- Metadata validation
- MIME type validation
"""

import unittest
from pydantic import ValidationError
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnemomatic.models import Document, Knowledge, Note


class TestDocumentValidation(unittest.TestCase):
    """Test Document input validation."""

    def test_document_valid(self):
        """Valid document should be created successfully."""
        doc = Document(
            namespace="test",
            title="Test Document",
            content="This is a test document.",
        )
        self.assertEqual(doc.namespace, "test")
        self.assertEqual(doc.title, "Test Document")

    def test_document_empty_namespace(self):
        """Empty namespace should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(namespace="", title="Test", content="Content")
        self.assertIn("namespace", str(cm.exception))

    def test_document_whitespace_only_namespace(self):
        """Whitespace-only namespace should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(namespace="   ", title="Test", content="Content")
        self.assertIn("namespace", str(cm.exception))

    def test_document_empty_title(self):
        """Empty title should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(namespace="test", title="", content="Content")
        self.assertIn("title", str(cm.exception))

    def test_document_whitespace_only_title(self):
        """Whitespace-only title should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(namespace="test", title="   \t\n", content="Content")
        self.assertIn("title", str(cm.exception))

    def test_document_empty_content(self):
        """Empty content should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(namespace="test", title="Test", content="")
        self.assertIn("content", str(cm.exception))

    def test_document_whitespace_only_content(self):
        """Whitespace-only content should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(namespace="test", title="Test", content="   \n\t  ")
        self.assertIn("content", str(cm.exception))

    def test_document_namespace_too_long(self):
        """Namespace exceeding max length should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="x" * 101,
                title="Test",
                content="Content"
            )
        self.assertIn("namespace", str(cm.exception))

    def test_document_title_too_long(self):
        """Title exceeding max length should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="x" * 501,
                content="Content"
            )
        self.assertIn("title", str(cm.exception))

    def test_document_content_too_long(self):
        """Content exceeding 100KB should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="x" * 100_001
            )
        self.assertIn("content", str(cm.exception))

    def test_document_invalid_mime_type(self):
        """MIME type without '/' should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                mime_type="invalid_type"
            )
        self.assertIn("mime_type", str(cm.exception))

    def test_document_valid_mime_types(self):
        """Various valid MIME types should work."""
        for mime_type in ["text/plain", "text/markdown", "application/json", "application/xml"]:
            doc = Document(
                namespace="test",
                title="Test",
                content="Content",
                mime_type=mime_type
            )
            self.assertEqual(doc.mime_type, mime_type)

    def test_document_empty_tag(self):
        """Empty tag in list should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                tags=["valid", "", "another"]
            )
        self.assertIn("tags", str(cm.exception))

    def test_document_whitespace_tag(self):
        """Whitespace-only tag should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                tags=["valid", "   ", "another"]
            )
        self.assertIn("tags", str(cm.exception))

    def test_document_tag_too_long(self):
        """Tag exceeding max length should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                tags=["x" * 51]
            )
        self.assertIn("tags", str(cm.exception))

    def test_document_too_many_tags(self):
        """More than 100 tags should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                tags=[f"tag{i}" for i in range(101)]
            )
        self.assertIn("tags", str(cm.exception))

    def test_document_non_string_tags(self):
        """Non-string tags should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                tags=["valid", 123, "another"]
            )
        self.assertIn("tags", str(cm.exception))

    def test_document_invalid_metadata_keys(self):
        """Non-string metadata keys should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                metadata={123: "value"}
            )
        self.assertIn("metadata", str(cm.exception))

    def test_document_empty_metadata_key(self):
        """Empty metadata key should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                metadata={"": "value"}
            )
        self.assertIn("metadata", str(cm.exception))

    def test_document_metadata_value_too_long(self):
        """Metadata value exceeding length should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                metadata={"key": "x" * 10_001}
            )
        self.assertIn("metadata", str(cm.exception))

    def test_document_too_many_metadata_keys(self):
        """More than 50 metadata keys should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Document(
                namespace="test",
                title="Test",
                content="Content",
                metadata={f"key{i}": f"value{i}" for i in range(51)}
            )
        self.assertIn("metadata", str(cm.exception))

    def test_document_valid_metadata(self):
        """Valid metadata should be accepted."""
        doc = Document(
            namespace="test",
            title="Test",
            content="Content",
            metadata={"author": "alice", "version": "1.0", "category": "tutorial"}
        )
        self.assertEqual(len(doc.metadata), 3)


class TestKnowledgeValidation(unittest.TestCase):
    """Test Knowledge input validation."""

    def test_knowledge_valid(self):
        """Valid knowledge entry should be created successfully."""
        k = Knowledge(
            namespace="test",
            subject="API endpoint",
            fact="Uses REST protocol on port 8000.",
            confidence=0.95,
            source="code-review"
        )
        self.assertEqual(k.subject, "API endpoint")
        self.assertEqual(k.confidence, 0.95)

    def test_knowledge_empty_subject(self):
        """Empty subject should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Knowledge(namespace="test", subject="", fact="A fact")
        self.assertIn("subject", str(cm.exception))

    def test_knowledge_empty_fact(self):
        """Empty fact should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Knowledge(namespace="test", subject="Subject", fact="")
        self.assertIn("fact", str(cm.exception))

    def test_knowledge_whitespace_fact(self):
        """Whitespace-only fact should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Knowledge(namespace="test", subject="Subject", fact="   \n")
        self.assertIn("fact", str(cm.exception))

    def test_knowledge_fact_too_long(self):
        """Fact exceeding 5000 chars should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Knowledge(
                namespace="test",
                subject="Subject",
                fact="x" * 5_001
            )
        self.assertIn("fact", str(cm.exception))

    def test_knowledge_confidence_below_zero(self):
        """Confidence below 0.0 should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Knowledge(
                namespace="test",
                subject="Subject",
                fact="A fact",
                confidence=-0.1
            )
        self.assertIn("confidence", str(cm.exception))

    def test_knowledge_confidence_above_one(self):
        """Confidence above 1.0 should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Knowledge(
                namespace="test",
                subject="Subject",
                fact="A fact",
                confidence=1.1
            )
        self.assertIn("confidence", str(cm.exception))

    def test_knowledge_confidence_at_bounds(self):
        """Confidence at exactly 0.0 and 1.0 should be accepted."""
        k1 = Knowledge(namespace="test", subject="S", fact="F", confidence=0.0)
        k2 = Knowledge(namespace="test", subject="S", fact="F", confidence=1.0)
        self.assertEqual(k1.confidence, 0.0)
        self.assertEqual(k2.confidence, 1.0)

    def test_knowledge_confidence_non_numeric(self):
        """Non-numeric confidence should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Knowledge(
                namespace="test",
                subject="Subject",
                fact="A fact",
                confidence="high"
            )
        self.assertIn("confidence", str(cm.exception))

    def test_knowledge_subject_too_long(self):
        """Subject exceeding 500 chars should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Knowledge(
                namespace="test",
                subject="x" * 501,
                fact="A fact"
            )
        self.assertIn("subject", str(cm.exception))

    def test_knowledge_source_too_long(self):
        """Source exceeding 100 chars should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Knowledge(
                namespace="test",
                subject="Subject",
                fact="A fact",
                source="x" * 101
            )
        self.assertIn("source", str(cm.exception))

    def test_knowledge_valid_confidence_values(self):
        """Various confidence values should be accepted."""
        for confidence in [0.0, 0.5, 0.75, 0.99, 1.0]:
            k = Knowledge(
                namespace="test",
                subject="S",
                fact="F",
                confidence=confidence
            )
            self.assertEqual(k.confidence, confidence)


class TestNoteValidation(unittest.TestCase):
    """Test Note input validation."""

    def test_note_valid(self):
        """Valid note should be created successfully."""
        note = Note(
            namespace="personal",
            title="Meeting Notes",
            content="Discussed project timeline",
            source="voice"
        )
        self.assertEqual(note.title, "Meeting Notes")
        self.assertEqual(note.source, "voice")

    def test_note_empty_title(self):
        """Empty title should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Note(namespace="test", title="", content="Content")
        self.assertIn("title", str(cm.exception))

    def test_note_empty_content(self):
        """Empty content should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Note(namespace="test", title="Title", content="")
        self.assertIn("content", str(cm.exception))

    def test_note_content_too_long(self):
        """Content exceeding 100KB should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Note(
                namespace="test",
                title="Title",
                content="x" * 100_001
            )
        self.assertIn("content", str(cm.exception))

    def test_note_source_too_long(self):
        """Source exceeding 100 chars should be rejected."""
        with self.assertRaises(ValidationError) as cm:
            Note(
                namespace="test",
                title="Title",
                content="Content",
                source="x" * 101
            )
        self.assertIn("source", str(cm.exception))

    def test_note_valid_sources(self):
        """Various source values should be accepted."""
        for source in ["text", "voice", "clipboard", "api", "user-input"]:
            note = Note(
                namespace="test",
                title="Title",
                content="Content",
                source=source
            )
            self.assertEqual(note.source, source)


class TestValidationEdgeCases(unittest.TestCase):
    """Test edge cases in validation."""

    def test_unicode_content(self):
        """Unicode content should be accepted."""
        doc = Document(
            namespace="test",
            title="Unicode Test 日本語",
            content="Hello 世界! 🚀 Привет"
        )
        self.assertIn("日本語", doc.title)

    def test_very_long_valid_content(self):
        """Content at max length should be accepted."""
        doc = Document(
            namespace="test",
            title="Test",
            content="x" * 100_000
        )
        self.assertEqual(len(doc.content), 100_000)

    def test_max_tags(self):
        """Exactly 100 tags should be accepted."""
        doc = Document(
            namespace="test",
            title="Test",
            content="Content",
            tags=[f"tag{i}" for i in range(100)]
        )
        self.assertEqual(len(doc.tags), 100)

    def test_special_characters_in_strings(self):
        """Special characters should be accepted."""
        doc = Document(
            namespace="test",
            title="Test: Special <chars> & symbols \"quotes\"",
            content="Line 1\nLine 2\tTabbed\r\nCRLF"
        )
        self.assertIn("Special", doc.title)

    def test_metadata_with_various_value_types(self):
        """Metadata can have various value types."""
        doc = Document(
            namespace="test",
            title="Test",
            content="Content",
            metadata={
                "string": "value",
                "number": 42,
                "float": 3.14,
                "bool": True,
                "null": None,
                "list": [1, 2, 3],
                "dict": {"nested": "value"}
            }
        )
        self.assertEqual(len(doc.metadata), 7)


if __name__ == "__main__":
    unittest.main()
