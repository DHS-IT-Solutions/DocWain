"""Auto-trigger pipeline orchestrator with tiered fast path and progress cards."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from teams_app.config import TeamsAppConfig
from teams_app.pipeline.fast_path import Pipeline, classify_file, should_escalate
from teams_app.pipeline.workers import WorkerPool
from teams_app.signals.capture import SignalCapture
from teams_app.storage.tenant import TenantManager

logger = logging.getLogger(__name__)


class TeamsAutoOrchestrator:
    """Orchestrates document processing with auto-trigger and tiered fast path.

    Wraps the existing src.teams.pipeline.TeamsDocumentPipeline stages,
    adding express/full routing, concurrent workers, and progress card updates.
    """

    def __init__(
        self,
        storage: Any,
        state_store: Any,
        tenant_manager: TenantManager,
        signal_capture: SignalCapture,
        config: Optional[TeamsAppConfig] = None,
    ):
        self.storage = storage
        self.state_store = state_store
        self.tenant_manager = tenant_manager
        self.signal_capture = signal_capture
        self.config = config or TeamsAppConfig()
        self.worker_pool = WorkerPool(max_concurrent=self.config.max_concurrent_documents)

    def classify(self, filename: str) -> Pipeline:
        """Classify a file for express or full pipeline."""
        return classify_file(filename)

    def should_escalate(self, extracted_text: str) -> bool:
        """Check if express extraction needs escalation to full pipeline."""
        return should_escalate(extracted_text, min_chars=self.config.express_min_chars)

    async def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        context: Any,
        turn_context: Any,
        correlation_id: str,
        auth_token: str = "",
    ) -> Dict[str, Any]:
        """Process a single document through the appropriate pipeline.

        Uses the existing src.teams.pipeline stages but controls routing.
        """
        from src.teams.pipeline import TeamsDocumentPipeline
        from src.teams.cards import build_card

        pipeline_type = self.classify(filename)
        started = time.monotonic()

        base_pipeline = TeamsDocumentPipeline(
            storage=self.storage,
            state_store=self.state_store,
        )

        # Send initial progress card
        progress_card = build_card("stage_progress_card",
            filename=filename,
            pipeline_type=pipeline_type.value,
            stage="downloading",
            progress_pct="10",
        )
        msg_id = await self._send_card(turn_context, progress_card)

        try:
            # Stage 1: Extraction
            await self._update_progress(turn_context, msg_id, filename, pipeline_type, "extracting", "25")

            if pipeline_type == Pipeline.EXPRESS:
                result = await self._express_extract(file_bytes, filename, context, correlation_id)
                extracted_text = result.get("extracted_text", "")

                if self.should_escalate(extracted_text):
                    logger.info("Escalating %s from express to full pipeline (text too short)", filename)
                    pipeline_type = Pipeline.FULL
                    result = await base_pipeline.stage_identify(
                        file_bytes, filename, content_type, context, correlation_id,
                    )
            else:
                result = await base_pipeline.stage_identify(
                    file_bytes, filename, content_type, context, correlation_id,
                )

            if not result:
                await self._update_progress(turn_context, msg_id, filename, pipeline_type, "failed", "0")
                return {"status": "failed", "filename": filename, "error": "Extraction failed"}

            # Stage 2: Screening
            await self._update_progress(turn_context, msg_id, filename, pipeline_type, "screening", "50")

            screen_result = await base_pipeline.stage_screen(
                document_id=result.get("document_id", ""),
                extracted_text=result.get("extracted_text", "")[:50000],
                filename=filename,
                correlation_id=correlation_id,
            )

            if screen_result and screen_result.get("needs_consent"):
                await self._update_progress(turn_context, msg_id, filename, pipeline_type, "blocked", "50")
                return {"status": "blocked", "filename": filename, "screen_result": screen_result}

            # Stage 3: Embedding
            await self._update_progress(turn_context, msg_id, filename, pipeline_type, "embedding", "75")

            embed_result = await base_pipeline.stage_embed(
                document_id=result.get("document_id", ""),
                extracted_content=result.get("extracted_docs", result),
                filename=filename,
                doc_type=result.get("doc_type", "unknown"),
                context=context,
                correlation_id=correlation_id,
                suggested_questions=result.get("suggested_questions", []),
                doc_intelligence=result.get("doc_intelligence"),
            )

            # Stage 4: KG (full pipeline only)
            if pipeline_type == Pipeline.FULL:
                await self._update_progress(turn_context, msg_id, filename, pipeline_type, "kg_building", "90")

            elapsed = time.monotonic() - started
            await self._update_progress(
                turn_context, msg_id, filename, pipeline_type, "ready",
                "100", elapsed_s=f"{elapsed:.1f}",
                sections=str(result.get("sections_count", 0)),
                chunks=str(embed_result.get("chunks_count", 0) if embed_result else 0),
            )

            return {
                "status": "ready",
                "filename": filename,
                "pipeline": pipeline_type.value,
                "elapsed_s": elapsed,
                "result": result,
                "embed_result": embed_result,
            }

        except Exception as exc:
            logger.exception("Pipeline failed for %s: %s", filename, exc)
            await self._update_progress(turn_context, msg_id, filename, pipeline_type, "failed", "0")
            return {"status": "failed", "filename": filename, "error": str(exc)}

    async def process_attachments(
        self,
        attachments: List[Dict[str, Any]],
        context: Any,
        turn_context: Any,
        correlation_id: str,
        auth_token: str = "",
    ) -> Dict[str, Any]:
        """Process multiple attachments concurrently via the worker pool."""
        tasks = []
        for att in attachments:
            file_bytes = att.get("file_bytes", b"")
            filename = att.get("filename", "unknown")
            content_type = att.get("content_type", "application/octet-stream")

            tasks.append((
                filename,
                self.process_document,
                (file_bytes, filename, content_type, context, turn_context, correlation_id, auth_token),
            ))

        return await self.worker_pool.run_all(tasks)

    async def _express_extract(
        self,
        file_bytes: bytes,
        filename: str,
        context: Any,
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Express extraction using native parsers only — no OCR/layout."""
        from src.api.dataHandler import fileProcessor

        extracted_docs = await asyncio.to_thread(fileProcessor, file_bytes, filename)
        if not extracted_docs:
            return {"extracted_text": "", "extracted_docs": {}}

        all_text = ""
        for doc_data in extracted_docs.values():
            if isinstance(doc_data, dict):
                all_text += doc_data.get("full_text", "") or doc_data.get("text", "")
            elif isinstance(doc_data, str):
                all_text += doc_data

        return {
            "extracted_text": all_text,
            "extracted_docs": extracted_docs,
            "document_id": correlation_id,
        }

    async def _send_card(self, turn_context: Any, card: Dict) -> Optional[str]:
        """Send an Adaptive Card and return the message ID for updates."""
        try:
            from botbuilder.schema import Activity, ActivityTypes, Attachment

            activity = Activity(
                type=ActivityTypes.message,
                attachments=[Attachment(
                    content_type="application/vnd.microsoft.card.adaptive",
                    content=card,
                )],
            )
            response = await turn_context.send_activity(activity)
            return response.id if response else None
        except Exception as exc:
            logger.error("Failed to send card: %s", exc)
            return None

    async def _update_progress(
        self,
        turn_context: Any,
        message_id: Optional[str],
        filename: str,
        pipeline_type: Pipeline,
        stage: str,
        progress_pct: str,
        **kwargs: str,
    ) -> None:
        """Update the in-place progress card."""
        if not message_id:
            return

        try:
            from botbuilder.schema import Activity, ActivityTypes, Attachment
            from src.teams.cards import build_card

            card = build_card("stage_progress_card",
                filename=filename,
                pipeline_type=pipeline_type.value,
                stage=stage,
                progress_pct=progress_pct,
                **kwargs,
            )
            update = Activity(
                id=message_id,
                type=ActivityTypes.message,
                attachments=[Attachment(
                    content_type="application/vnd.microsoft.card.adaptive",
                    content=card,
                )],
            )
            await turn_context.update_activity(update)
        except Exception as exc:
            logger.debug("Progress card update failed (non-fatal): %s", exc)
