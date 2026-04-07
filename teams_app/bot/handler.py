"""Standalone Teams bot handler — extends src.teams.bot_app with auto-trigger pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

from botbuilder.core import TurnContext
from botbuilder.schema import Activity, ActivityTypes, Attachment

from src.teams.bot_app import DocWainTeamsBot
from src.teams.logic import TeamsChatService, TeamsChatContext
from src.teams.state import TeamsStateStore
from src.teams.tools import TeamsToolRouter, format_text_answer
from src.teams.cards import build_card
# _resolve_download_url and _resolve_filename expect raw dicts (Dict[str, Any]), not
# Bot Framework Attachment objects — callers must use att.as_dict() first.
from src.teams.attachments import _resolve_auth_token, _resolve_download_url, _resolve_filename

from teams_app.config import TeamsAppConfig
from teams_app.graph.onedrive import is_onedrive_url, download_shared_file
from teams_app.pipeline.orchestrator import TeamsAutoOrchestrator
from teams_app.proxy.query_handler import TeamsQueryHandler
from teams_app.signals.capture import SignalCapture
from teams_app.storage.tenant import TenantManager

logger = logging.getLogger(__name__)


class StandaloneTeamsBot(DocWainTeamsBot):
    """Extends DocWainTeamsBot with auto-trigger pipeline and query proxy.

    Overrides on_message_activity to:
    1. Auto-provision tenants from AAD identity
    2. Route file attachments through the auto-trigger orchestrator
    3. Detect OneDrive/SharePoint links and download via Graph API
    4. Proxy text queries to the main app instead of local RAG
    5. Capture learning signals for finetuning
    """

    def __init__(
        self,
        orchestrator: TeamsAutoOrchestrator,
        query_handler: TeamsQueryHandler,
        tenant_manager: TenantManager,
        signal_capture: SignalCapture,
        config: Optional[TeamsAppConfig] = None,
    ):
        # DocWainTeamsBot.__init__ takes no arguments beyond self — it binds
        # module-level singletons (chat_service, state_store, tool_router).
        super().__init__()
        self.orchestrator = orchestrator
        self.query_handler = query_handler
        self.tenant_manager = tenant_manager
        self.signal_capture = signal_capture
        self.config = config or TeamsAppConfig()

    async def on_message_activity(self, turn_context: TurnContext) -> None:
        """Handle incoming Teams messages — auto-trigger pipeline or proxy query."""
        activity = turn_context.activity
        correlation_id = str(uuid.uuid4())[:8]

        # Auto-provision tenant & user
        raw_activity = activity.as_dict() if hasattr(activity, "as_dict") else {}
        tenant_id, user_id, display_name = TenantManager.extract_identity(raw_activity)

        if tenant_id:
            self.tenant_manager.ensure_tenant(tenant_id, display_name)
            self.tenant_manager.ensure_user(user_id, tenant_id, display_name)

        # build_context is a @staticmethod on TeamsChatService; accessible via instance too.
        context = self.chat_service.build_context(
            user_id=user_id,
            session_id=activity.conversation.id if activity.conversation else user_id,
        )

        # Check for file attachments
        attachments = self._extract_file_attachments(activity)

        if attachments:
            auth_token = await _resolve_auth_token(turn_context, None)
            await self._handle_attachments(
                attachments, context, turn_context, correlation_id, auth_token, tenant_id,
            )
            return

        # Check for Adaptive Card actions
        if activity.value and isinstance(activity.value, dict):
            action = activity.value.get("action", "")

            # Handle feedback from response cards
            if action == "feedback":
                signal = activity.value.get("signal", "implicit")
                query = activity.value.get("query", "")
                self.signal_capture.record(
                    query=query, response="", sources=[], grounded=True,
                    context_found=True, signal=signal, tenant_id=tenant_id or "",
                )
                emoji = "\ud83d\udc4d" if signal == "positive" else "\ud83d\udc4e"
                await turn_context.send_activity(f"Thanks for the feedback! {emoji}")
                return

            # Route query-based actions through the proxy (not local RAG)
            if action in ("domain_query", "summarize_recent"):
                query = activity.value.get("query", "Summarize this document")
                await self._handle_query(query, context, turn_context, tenant_id)
                return

            # Non-query actions (delete, preferences, tools menu) use the tool router
            # but skip any that would trigger local RAG
            # Clear documents action
            if action in ("delete_documents", "confirm_delete"):
                await self._clear_documents(context, turn_context, tenant_id)
                return

            if action in ("tools", "show_preferences",
                          "set_model", "set_persona", "list_docs"):
                try:
                    result = await self.tool_router.handle_action(activity.value, context)
                    if result:
                        await turn_context.send_activity(Activity(**result))
                except Exception as exc:
                    logger.error("Tool action '%s' failed: %s", action, exc)
                    await turn_context.send_activity(f"Action failed: {exc}")
                return

            # Unknown action — try proxy as a query
            query = activity.value.get("query", activity.value.get("text", ""))
            if query:
                await self._handle_query(query, context, turn_context, tenant_id)
                return

            return

        # Check for OneDrive/SharePoint links in text
        text = (activity.text or "").strip()
        onedrive_url = is_onedrive_url(text)
        if onedrive_url:
            await self._handle_onedrive_link(
                onedrive_url, context, turn_context, correlation_id, tenant_id,
            )
            return

        # Handle text queries — proxy to main app
        if text:
            lower = text.lower()
            if lower in ("help", "/help"):
                card = build_card("help_card")
                await self._send_card_activity(turn_context, card)
                return
            if lower in ("tools", "/tools"):
                result = await self.tool_router.handle_action({"action": "tools"}, context)
                if result:
                    await turn_context.send_activity(Activity(**result))
                return
            if any(phrase in lower for phrase in ("delete all", "remove all", "clear all", "clear documents", "reset")):
                await self._clear_documents(context, turn_context, tenant_id)
                return

            # Check for refresh command
            if lower.startswith("refresh "):
                await turn_context.send_activity("Refresh is not yet implemented for this document. Please re-upload the file.")
                return

            await self._handle_query(text, context, turn_context, tenant_id)
            return

        await turn_context.send_activity("Send me a document to analyze, or ask a question about your uploaded documents.")

    def _extract_file_attachments(self, activity: Activity) -> list:
        """Extract file attachments from the activity."""
        if not activity.attachments:
            return []

        file_attachments = []
        for att in activity.attachments:
            content_type = att.content_type or ""
            if "card" in content_type.lower() or "adaptive" in content_type.lower():
                continue
            if att.name or (att.content_url and att.content_type):
                file_attachments.append(att)
        return file_attachments

    async def _handle_attachments(
        self,
        attachments: list,
        context: TeamsChatContext,
        turn_context: TurnContext,
        correlation_id: str,
        auth_token: str,
        tenant_id: str,
    ) -> None:
        """Fast document pipeline with clear progress cards.

        Skips the slow DI/cloud LLM stage — goes straight to:
        extract → screen → embed with in-place card updates.
        """
        import asyncio
        from teams_app.pipeline.cards import (
            progress_card, intelligence_card, error_card,
            send_card, update_card,
        )
        from teams_app.pipeline.fast_path import classify_file, Pipeline
        from src.teams.attachments import (
            _resolve_download_url, _resolve_filename,
            _download_bytes, _build_download_headers,
            _run_security_screening,
        )
        from src.api.dataHandler import fileProcessor
        from src.teams.pipeline import _build_teams_collection_name
        from src.api.config import Config
        from teams_app.pipeline.embedder import embed_document
        from teams_app.pipeline.intelligence import analyze_document

        for att in attachments:
            att_dict = att.as_dict() if hasattr(att, "as_dict") else {
                "contentType": getattr(att, "content_type", None),
                "name": getattr(att, "name", None),
                "contentUrl": getattr(att, "content_url", None),
                "content": getattr(att, "content", None),
            }
            filename = _resolve_filename(att_dict)
            download_url = _resolve_download_url(att_dict)

            if not download_url:
                await turn_context.send_activity(f"Could not resolve download URL for {filename}.")
                continue

            pipeline_type = classify_file(filename)

            # Step 1: Download — send initial progress card
            card = progress_card(filename, "1/5", "Downloading file...", 5)
            card_id = await send_card(turn_context, card)

            try:
                headers = _build_download_headers(auth_token)
                file_bytes = await _download_bytes(
                    download_url, headers=headers,
                    timeout=float(getattr(Config.Teams, "HTTP_TIMEOUT_SEC", 20)),
                    retries=int(getattr(Config.Teams, "HTTP_RETRIES", 2)),
                    max_bytes=int(getattr(Config.Teams, "MAX_ATTACHMENT_MB", 50)) * 1024 * 1024,
                )
            except Exception as exc:
                logger.error("Download failed for %s: %s", filename, exc)
                await update_card(turn_context, card_id, error_card(filename, f"Download failed: {exc}"))
                continue

            # Step 2: Extract
            card = progress_card(filename, "2/5", f"Extracting content ({pipeline_type.value} pipeline)...", 20)
            await update_card(turn_context, card_id, card)

            try:
                extracted_docs = await asyncio.to_thread(fileProcessor, file_bytes, filename)
                if not extracted_docs:
                    await update_card(turn_context, card_id, error_card(filename, "No extractable content found."))
                    continue

                # Get extracted text for screening and intelligence
                all_text = ""
                for doc_data in extracted_docs.values():
                    if isinstance(doc_data, dict):
                        # Try full_text first, then texts list, then text field
                        ft = doc_data.get("full_text") or doc_data.get("text") or doc_data.get("content") or ""
                        if ft:
                            all_text += ft + "\n\n"
                        elif doc_data.get("texts"):
                            for t in doc_data["texts"]:
                                if isinstance(t, str):
                                    all_text += t + "\n"
                                elif isinstance(t, dict):
                                    all_text += (t.get("text") or t.get("content") or "") + "\n"
                    elif isinstance(doc_data, str):
                        all_text += doc_data + "\n\n"
                    elif hasattr(doc_data, "full_text"):
                        all_text += (getattr(doc_data, "full_text", "") or "") + "\n\n"

                all_text = all_text.strip()
                logger.info("Extracted %d chars from %s (%d docs)", len(all_text), filename, len(extracted_docs))
            except Exception as exc:
                logger.error("Extraction failed for %s: %s", filename, exc)
                await update_card(turn_context, card_id, error_card(filename, f"Extraction failed: {exc}"))
                continue

            # Step 3: Screen
            card = progress_card(filename, "3/5", "Running security screening...", 40)
            await update_card(turn_context, card_id, card)

            try:
                from src.utils.logging_utils import get_logger as _get_logger
                _log = _get_logger(__name__, correlation_id)
                screen_result = await asyncio.to_thread(
                    _run_security_screening, all_text[:50000], filename, correlation_id, _log
                )
                risk_level = screen_result.risk_level if screen_result else "LOW"
            except Exception as exc:
                logger.warning("Screening failed for %s (proceeding): %s", filename, exc)
                risk_level = "UNKNOWN"

            # Step 4: Embed
            card = progress_card(filename, "4/5", f"Security: {risk_level} risk. Embedding for retrieval...", 60)
            await update_card(turn_context, card_id, card)

            try:
                collection_name = _build_teams_collection_name(
                    context.subscription_id, context.profile_id,
                )

                chunks_count, quality_grade, doc_type = await embed_document(
                    extracted_docs=extracted_docs,
                    filename=filename,
                    doc_tag=correlation_id,
                    collection_name=collection_name,
                    profile_id=context.profile_id,
                )

                # Record upload in state store
                self.orchestrator.state_store.record_upload(
                    context.subscription_id,
                    context.profile_id,
                    filename,
                    correlation_id,
                    chunks_count,
                    document_type=doc_type,
                )

            except Exception as exc:
                logger.error("Embedding failed for %s: %s", filename, exc, exc_info=True)
                await update_card(turn_context, card_id, error_card(filename, f"Embedding failed: {exc}"))
                continue

            # Step 5: Intelligence analysis via LLM — generate summary + 5 smart questions
            card = progress_card(filename, "5/5", "Analyzing document intelligence...", 90)
            await update_card(turn_context, card_id, card)

            intel = await analyze_document(all_text, filename, fallback_doc_type=doc_type)
            # Use LLM-detected doc_type if available (overrides heuristic)
            if intel.doc_type and intel.doc_type != "general":
                doc_type = intel.doc_type

            # Show intelligence card with summary + 5 clickable question buttons
            done_card = intelligence_card(
                filename=filename,
                chunks_count=chunks_count,
                quality_grade=quality_grade,
                doc_type=doc_type,
                summary=intel.summary,
                key_entities=intel.key_entities,
                questions=intel.questions,
            )
            await update_card(turn_context, card_id, done_card)
            logger.info("Pipeline complete for %s: %d chunks, grade=%s, type=%s, entities=%s",
                         filename, chunks_count, quality_grade, doc_type, intel.key_entities[:3])

    async def _clear_documents(
        self,
        context: TeamsChatContext,
        turn_context: TurnContext,
        tenant_id: str,
    ) -> None:
        """Clear all documents from the Teams Qdrant collection and state."""
        try:
            from src.teams.pipeline import _build_teams_collection_name
            from qdrant_client import QdrantClient
            from src.api.config import Config

            collection_name = _build_teams_collection_name(
                context.subscription_id, context.profile_id,
            )

            client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=30)

            # Delete the entire collection
            try:
                client.delete_collection(collection_name)
                logger.info("Deleted Qdrant collection %s", collection_name)
            except Exception:
                pass  # Collection may not exist

            # Clear state store uploads
            try:
                self.orchestrator.state_store.clear_uploads(
                    context.subscription_id, context.profile_id,
                )
            except Exception:
                pass

            await turn_context.send_activity(
                "All documents and memory cleared. Upload new documents to get started fresh."
            )
            logger.info("Documents cleared for collection %s", collection_name)

        except Exception as exc:
            logger.error("Clear documents failed: %s", exc)
            await turn_context.send_activity(f"Failed to clear documents: {exc}")

    async def _handle_onedrive_link(
        self,
        url: str,
        context: TeamsChatContext,
        turn_context: TurnContext,
        correlation_id: str,
        tenant_id: str,
    ) -> None:
        """Download a file from OneDrive/SharePoint and process it."""
        await turn_context.send_activity("Downloading from OneDrive/SharePoint...")
        try:
            auth_token = await _resolve_auth_token(turn_context, None)
            file_bytes, filename = await download_shared_file(url, auth_token)
            await self.orchestrator.process_attachments(
                [{"file_bytes": file_bytes, "filename": filename, "content_type": "application/octet-stream"}],
                context, turn_context, correlation_id, auth_token,
            )
        except Exception as exc:
            logger.error("OneDrive download failed: %s", exc)
            await turn_context.send_activity(f"Failed to download from OneDrive: {exc}")

    async def _handle_query(
        self,
        query: str,
        context: TeamsChatContext,
        turn_context: TurnContext,
        tenant_id: str,
    ) -> None:
        """Search Qdrant directly and generate a response via LLM."""
        started = time.monotonic()

        await turn_context.send_activities([Activity(type=ActivityTypes.typing)])

        from src.teams.pipeline import _build_teams_collection_name
        collection_name = _build_teams_collection_name(
            context.subscription_id, context.profile_id,
        )

        result = await self.query_handler.answer(
            query=query,
            collection_name=collection_name,
            user_id=context.user_id,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        if not result.context_found:
            try:
                has_docs = self.orchestrator.storage.get_document_count(tenant_id) > 0 if tenant_id else False
            except Exception:
                has_docs = False
            if not has_docs:
                result.response += "\n\n_I don't have any documents to search yet. Send me a file to get started!_"

        # Build response card with feedback buttons
        card = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": result.response, "wrap": True},
            ],
            "actions": [
                {
                    "type": "Action.Submit",
                    "title": "\ud83d\udc4d",
                    "data": {"action": "feedback", "signal": "positive", "query": query},
                },
                {
                    "type": "Action.Submit",
                    "title": "\ud83d\udc4e",
                    "data": {"action": "feedback", "signal": "negative", "query": query},
                },
            ],
        }

        if result.sources:
            sources_text = "\n".join(f"- {s.get('title', 'Unknown')}" for s in result.sources[:5])
            card["body"].append({"type": "TextBlock", "text": f"**Sources:**\n{sources_text}", "wrap": True, "size": "Small"})

        await self._send_card_activity(turn_context, card)

        self.signal_capture.record(
            query=query,
            response=result.response,
            sources=result.sources,
            grounded=result.grounded,
            context_found=result.context_found,
            signal="implicit",
            tenant_id=tenant_id or "",
            latency_ms=elapsed_ms,
        )

    async def _download_file(self, url: str, auth_token: str) -> bytes:
        """Download a file from a URL with auth token."""
        import httpx
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            return resp.content

    async def _send_card_activity(self, turn_context: TurnContext, card: Dict) -> Optional[str]:
        """Send an Adaptive Card and return the message ID."""
        activity = Activity(
            type=ActivityTypes.message,
            attachments=[Attachment(
                content_type="application/vnd.microsoft.card.adaptive",
                content=card,
            )],
        )
        response = await turn_context.send_activity(activity)
        return response.id if response else None

    async def on_members_added_activity(self, members_added, turn_context: TurnContext) -> None:
        """Send welcome card with onboarding message when bot is added."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                card = build_card("welcome_card")
                await self._send_card_activity(turn_context, card)
