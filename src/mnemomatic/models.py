import json
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

# Validation constants
MAX_NAMESPACE_LENGTH = 100
MAX_TITLE_LENGTH = 500
MAX_SUBJECT_LENGTH = 500
MAX_CONTENT_LENGTH = 100_000  # 100 KB
MAX_FACT_LENGTH = 5_000  # Facts should be shorter than documents
MAX_SOURCE_LENGTH = 100
MAX_TAGS = 100
MAX_TAG_LENGTH = 50
MAX_METADATA_KEYS = 50
MAX_METADATA_KEY_LENGTH = 200
MAX_METADATA_VALUE_LENGTH = 10_000  # measured on the JSON-serialized value


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


def _validate_whitespace(v):
    if isinstance(v, str) and v.strip() == "":
        raise ValueError("must not be empty or contain only whitespace")
    return v


def _validate_tags(v):
    if not isinstance(v, list):
        raise ValueError("tags must be a list")
    for tag in v:
        if not isinstance(tag, str):
            raise ValueError(f"all tags must be strings, got {type(tag).__name__}")
        if not tag or tag.strip() == "":
            raise ValueError("tags must not be empty or whitespace-only")
        if len(tag) > MAX_TAG_LENGTH:
            raise ValueError(f"tag exceeds max length {MAX_TAG_LENGTH}: {tag[:20]}...")
    return v


def _validate_metadata(v):
    if not isinstance(v, dict):
        raise ValueError("metadata must be a dict")
    if len(v) > MAX_METADATA_KEYS:
        raise ValueError(f"metadata exceeds max keys {MAX_METADATA_KEYS}")
    for key, value in v.items():
        if not isinstance(key, str):
            raise ValueError(f"metadata keys must be strings, got {type(key).__name__}")
        if not key:
            raise ValueError("metadata keys must not be empty")
        if len(key) > MAX_METADATA_KEY_LENGTH:
            raise ValueError(f"metadata key exceeds max length {MAX_METADATA_KEY_LENGTH}: {key[:20]}...")
        # Size-check the serialized form so nested lists/dicts are bounded too,
        # and reject anything json.dumps can't store (it would fail at the DB
        # layer otherwise). RecursionError guards absurdly deep nesting.
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError, RecursionError):
            raise ValueError(f"metadata value for key {key!r} is not JSON-serializable")
        if len(serialized) > MAX_METADATA_VALUE_LENGTH:
            raise ValueError(
                f"metadata value for key {key!r} exceeds max serialized length {MAX_METADATA_VALUE_LENGTH}"
            )
    return v


class _MemoryItem(BaseModel):
    """Fields and validation shared by the three content types.

    Subclasses add their content fields and validate those for whitespace
    themselves — the base can only vouch for the fields it declares.
    """

    id: str = Field(default_factory=_new_id)
    namespace: str = Field(min_length=1, max_length=MAX_NAMESPACE_LENGTH)
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    # Usage tracking, maintained by Database.record_access — read-only from
    # the API's perspective; never written through the store/update paths.
    retrieval_count: int = 0
    last_accessed: datetime | None = None

    @field_validator("namespace", mode="before")
    @classmethod
    def validate_namespace_not_blank(cls, v):
        return _validate_whitespace(v)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        return _validate_tags(v)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        return _validate_metadata(v)


class Document(_MemoryItem):
    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_LENGTH)
    mime_type: str = "text/markdown"

    @field_validator("title", "content", mode="before")
    @classmethod
    def validate_no_only_whitespace(cls, v):
        return _validate_whitespace(v)

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v):
        """MIME type should be reasonable (basic check)."""
        if not isinstance(v, str) or not v:
            raise ValueError("mime_type must be a non-empty string")
        if "/" not in v:
            raise ValueError("mime_type must be in format 'type/subtype' (e.g., 'text/plain')")
        return v


class Knowledge(_MemoryItem):
    subject: str = Field(min_length=1, max_length=MAX_SUBJECT_LENGTH)
    fact: str = Field(min_length=1, max_length=MAX_FACT_LENGTH)
    confidence: float = 1.0
    source: str = Field(default="unknown", max_length=MAX_SOURCE_LENGTH)
    # Temporal validity, maintained by the store/supersede paths. A current
    # fact has both as None; a superseded one records when it stopped being
    # the current answer and which entry replaced it.
    valid_until: datetime | None = None
    superseded_by: str | None = None

    @field_validator("subject", "fact", "source", mode="before")
    @classmethod
    def validate_no_only_whitespace(cls, v):
        return _validate_whitespace(v)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Confidence must be between 0.0 and 1.0."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"confidence must be a number, got {type(v).__name__}")
        if v < 0.0 or v > 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return float(v)


class Note(_MemoryItem):
    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_LENGTH)
    source: str = Field(default="text", max_length=MAX_SOURCE_LENGTH)

    @field_validator("title", "content", "source", mode="before")
    @classmethod
    def validate_no_only_whitespace(cls, v):
        return _validate_whitespace(v)


class SearchResult(BaseModel):
    id: str
    type: str  # "document", "knowledge", or "note"
    namespace: str
    title: str  # title for documents/notes, subject for knowledge
    snippet: str  # content preview or fact
    resource_uri: str  # mnemomatic:// URI for full content retrieval
    score: float
    tags: list[str] = Field(default_factory=list)
    partial: bool = False  # True when snippet is a chunk; call read(id) for full content
