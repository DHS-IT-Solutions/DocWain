"""Tests for standalone API Pydantic schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.standalone_schemas import (
    AsyncAcceptedResponse,
    BatchItemResult,
    BatchRequest,
    BatchResponse,
    DocumentStatusResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    ExtractRequest,
    ExtractResponse,
    MultiProcessRequest,
    ProcessRequest,
    ProcessResponse,
    QueryRequest,
    SourceRef,
    StandaloneErrorDetail,
    StandaloneErrorResponse,
    TemplateInfo,
    TemplatesResponse,
    UsageResponse,
    UsageStats,
)


class TestProcessRequestDefaults:
    def test_process_request_defaults(self):
        req = ProcessRequest(prompt="What is the contract value?")
        assert req.prompt == "What is the contract value?"
        assert req.mode == "qa"
        assert req.output_format == "json"
        assert req.persist is False
        assert req.stream is False
        assert req.template is None
        assert req.confidence_threshold == 0.0
        assert req.callback_url is None

    def test_process_request_all_fields(self):
        req = ProcessRequest(
            prompt="List all clauses",
            mode="entities",
            output_format="markdown",
            persist=True,
            stream=True,
            template="legal_review",
            confidence_threshold=0.75,
            callback_url="https://example.com/webhook",
        )
        assert req.mode == "entities"
        assert req.output_format == "markdown"
        assert req.persist is True
        assert req.stream is True
        assert req.template == "legal_review"
        assert req.confidence_threshold == 0.75
        assert req.callback_url == "https://example.com/webhook"

    def test_process_request_validates_mode(self):
        with pytest.raises(ValidationError):
            ProcessRequest(prompt="test", mode="invalid_mode")

    def test_process_request_validates_output_format(self):
        with pytest.raises(ValidationError):
            ProcessRequest(prompt="test", output_format="xml")

    def test_process_request_confidence_threshold_bounds(self):
        req_min = ProcessRequest(prompt="test", confidence_threshold=0.0)
        assert req_min.confidence_threshold == 0.0
        req_max = ProcessRequest(prompt="test", confidence_threshold=1.0)
        assert req_max.confidence_threshold == 1.0
        with pytest.raises(ValidationError):
            ProcessRequest(prompt="test", confidence_threshold=1.1)
        with pytest.raises(ValidationError):
            ProcessRequest(prompt="test", confidence_threshold=-0.1)


class TestProcessResponseShape:
    def test_process_response_defaults(self):
        resp = ProcessResponse(request_id="req-001")
        assert resp.request_id == "req-001"
        assert resp.status == "completed"
        assert resp.answer is None
        assert resp.sources == []
        assert resp.confidence == 0.0
        assert resp.grounded is False
        assert resp.low_confidence is False
        assert resp.low_confidence_reasons == []
        assert resp.structured_output is None
        assert resp.document_id is None
        assert resp.output_format == "json"
        assert resp.partial_answer is None
        assert isinstance(resp.usage, UsageStats)

    def test_process_response_with_answer(self):
        resp = ProcessResponse(
            request_id="req-002",
            answer="The contract value is $1M",
            confidence=0.92,
            grounded=True,
            sources=[{"page": 3, "section": "2.1"}],
        )
        assert resp.answer == "The contract value is $1M"
        assert resp.confidence == 0.92
        assert resp.grounded is True
        assert len(resp.sources) == 1


class TestBatchResponseShape:
    def test_batch_response_shape(self):
        resp = BatchResponse(batch_id="batch-001", status="completed")
        assert resp.batch_id == "batch-001"
        assert resp.status == "completed"
        assert resp.results == []
        assert resp.summary == {}
        assert isinstance(resp.usage, UsageStats)

    def test_batch_response_with_results(self):
        item = BatchItemResult(
            filename="contract.pdf",
            status="completed",
            answer="Found 3 clauses",
            confidence=0.85,
        )
        resp = BatchResponse(
            batch_id="batch-002",
            status="completed",
            results=[item],
            summary={"total": 1, "succeeded": 1, "failed": 0},
        )
        assert len(resp.results) == 1
        assert resp.results[0].filename == "contract.pdf"
        assert resp.summary["total"] == 1

    def test_batch_item_result_defaults(self):
        item = BatchItemResult(filename="doc.pdf", status="completed")
        assert item.answer is None
        assert item.confidence == 0.0
        assert item.structured_output is None
        assert item.error is None


class TestExtractRequestRequiresMode:
    def test_extract_request_requires_mode(self):
        with pytest.raises(ValidationError):
            ExtractRequest()  # mode is required

    def test_extract_request_valid(self):
        req = ExtractRequest(mode="table")
        assert req.mode == "table"
        assert req.prompt is None
        assert req.output_format == "json"
        assert req.template is None

    def test_extract_request_all_modes(self):
        for mode in ("table", "entities", "summary"):
            req = ExtractRequest(mode=mode)
            assert req.mode == mode

    def test_extract_request_invalid_mode(self):
        with pytest.raises(ValidationError):
            ExtractRequest(mode="qa")  # "qa" not allowed in ExtractRequest


class TestQueryRequest:
    def test_query_request_requires_document_id(self):
        # Basic construction with just prompt should succeed
        req = QueryRequest(prompt="What are the key terms?")
        assert req.prompt == "What are the key terms?"
        assert req.document_id is None
        assert req.document_ids is None
        assert req.mode == "qa"
        assert req.output_format == "json"
        assert req.stream is False
        assert req.confidence_threshold == 0.0

    def test_query_request_with_document_id(self):
        req = QueryRequest(
            prompt="Summarize this document",
            document_id="doc-abc-123",
        )
        assert req.document_id == "doc-abc-123"

    def test_query_request_with_document_ids(self):
        req = QueryRequest(
            prompt="Compare these documents",
            document_ids=["doc-001", "doc-002"],
        )
        assert req.document_ids == ["doc-001", "doc-002"]


class TestUsageResponse:
    def test_usage_response(self):
        resp = UsageResponse(
            api_key_name="my-key",
            period="2026-04",
            totals={"requests": 100, "tokens": 50000},
            by_endpoint={"/process": 60, "/query": 40},
            by_mode={"qa": 80, "summary": 20},
            recent=[{"timestamp": "2026-04-08T10:00:00Z", "endpoint": "/process"}],
        )
        assert resp.api_key_name == "my-key"
        assert resp.period == "2026-04"
        assert resp.totals["requests"] == 100
        assert resp.by_endpoint["/process"] == 60
        assert resp.by_mode["qa"] == 80
        assert len(resp.recent) == 1


class TestUsageStats:
    def test_usage_stats_defaults(self):
        stats = UsageStats()
        assert stats.extraction_ms == 0
        assert stats.intelligence_ms == 0
        assert stats.retrieval_ms == 0
        assert stats.generation_ms == 0
        assert stats.total_ms == 0

    def test_usage_stats_with_values(self):
        stats = UsageStats(
            extraction_ms=100,
            intelligence_ms=200,
            retrieval_ms=50,
            generation_ms=300,
            total_ms=650,
        )
        assert stats.total_ms == 650


class TestSourceRef:
    def test_source_ref_all_optional(self):
        ref = SourceRef()
        assert ref.page is None
        assert ref.section is None
        assert ref.confidence is None
        assert ref.document is None
        assert ref.document_id is None

    def test_source_ref_with_values(self):
        ref = SourceRef(page=5, section="3.2", confidence=0.9, document="contract.pdf", document_id="doc-001")
        assert ref.page == 5
        assert ref.section == "3.2"


class TestMultiProcessRequest:
    def test_multi_process_request_defaults(self):
        req = MultiProcessRequest(prompt="Analyze all documents")
        assert req.prompt == "Analyze all documents"
        assert req.document_ids is None
        assert req.mode == "qa"
        assert req.output_format == "json"
        assert req.callback_url is None


class TestAsyncAcceptedResponse:
    def test_async_accepted_response_defaults(self):
        resp = AsyncAcceptedResponse(request_id="req-async-001")
        assert resp.request_id == "req-async-001"
        assert resp.status == "processing"
        assert resp.poll_url is None

    def test_async_accepted_response_with_poll_url(self):
        resp = AsyncAcceptedResponse(
            request_id="req-async-002",
            poll_url="/api/v1/status/req-async-002",
        )
        assert resp.poll_url == "/api/v1/status/req-async-002"


class TestDocumentUploadResponse:
    def test_document_upload_response(self):
        resp = DocumentUploadResponse(
            document_id="doc-xyz",
            created_at="2026-04-08T12:00:00Z",
        )
        assert resp.document_id == "doc-xyz"
        assert resp.status == "processing"
        assert resp.name is None
        assert resp.created_at == "2026-04-08T12:00:00Z"


class TestDocumentStatusResponse:
    def test_document_status_response(self):
        resp = DocumentStatusResponse(document_id="doc-001", status="ready")
        assert resp.document_id == "doc-001"
        assert resp.status == "ready"
        assert resp.name is None
        assert resp.pages is None
        assert resp.document_type is None
        assert resp.created_at is None
        assert resp.ready_at is None


class TestDocumentUploadRequest:
    def test_document_upload_request_optional_name(self):
        req = DocumentUploadRequest()
        assert req.name is None

    def test_document_upload_request_with_name(self):
        req = DocumentUploadRequest(name="my_contract.pdf")
        assert req.name == "my_contract.pdf"


class TestExtractResponse:
    def test_extract_response_defaults(self):
        resp = ExtractResponse(request_id="req-ext-001", mode="table")
        assert resp.request_id == "req-ext-001"
        assert resp.mode == "table"
        assert resp.result is None
        assert resp.metadata == {}


class TestTemplates:
    def test_template_info(self):
        t = TemplateInfo(name="legal_review", description="Legal document review", modes=["qa", "entities"])
        assert t.name == "legal_review"
        assert t.modes == ["qa", "entities"]

    def test_templates_response(self):
        resp = TemplatesResponse(
            templates=[
                TemplateInfo(name="legal_review", description="Legal review", modes=["qa"]),
                TemplateInfo(name="financial_summary", description="Financial summary", modes=["summary"]),
            ]
        )
        assert len(resp.templates) == 2


class TestStandaloneError:
    def test_standalone_error_detail(self):
        err = StandaloneErrorDetail(code="NOT_FOUND", message="Document not found")
        assert err.code == "NOT_FOUND"
        assert err.message == "Document not found"
        assert err.request_id is None

    def test_standalone_error_response(self):
        resp = StandaloneErrorResponse(
            error=StandaloneErrorDetail(
                code="VALIDATION_ERROR",
                message="Invalid mode",
                request_id="req-001",
            )
        )
        assert resp.error.code == "VALIDATION_ERROR"
        assert resp.error.request_id == "req-001"


class TestBatchRequest:
    def test_batch_request_defaults(self):
        req = BatchRequest(prompt="Extract key terms from each document")
        assert req.prompt == "Extract key terms from each document"
        assert req.mode == "qa"
        assert req.output_format == "json"
        assert req.callback_url is None
