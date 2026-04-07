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
from teams_app.proxy.query_proxy import QueryProxy, QueryRequest
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
        query_proxy: QueryProxy,
        tenant_manager: TenantManager,
        signal_capture: SignalCapture,
        config: Optional[TeamsAppConfig] = None,
    ):
        # DocWainTeamsBot.__init__ takes no arguments beyond self — it binds
        # module-level singletons (chat_service, state_store, tool_router).
        super().__init__()
        self.orchestrator = orchestrator
        self.query_proxy = query_proxy
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

            result = await self.tool_router.handle_action(activity.value, context)
            if result:
                await turn_context.send_activity(Activity(**result))
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
            if any(phrase in lower for phrase in ("delete all", "remove all", "clear all")):
                card = build_card("delete_confirm_card")
                await self._send_card_activity(turn_context, card)
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
        """Download and process file attachments through the auto-trigger orchestrator."""
        prepared = []
        for att in attachments:
            try:
                # _resolve_download_url and _resolve_filename expect raw dicts, not
                # Bot Framework Attachment objects — convert via as_dict() first.
                att_dict = att.as_dict() if hasattr(att, "as_dict") else {
                    "contentType": getattr(att, "content_type", None),
                    "name": getattr(att, "name", None),
                    "contentUrl": getattr(att, "content_url", None),
                }
                url = _resolve_download_url(att_dict)
                filename = _resolve_filename(att_dict)
                if not url:
                    await turn_context.send_activity(f"Could not resolve download URL for {filename}.")
                    continue
                file_bytes = await self._download_file(url, auth_token)
                prepared.append({
                    "file_bytes": file_bytes,
                    "filename": filename,
                    "content_type": att.content_type or "application/octet-stream",
                })
            except Exception as exc:
                logger.error("Failed to download attachment %s: %s", getattr(att, "name", "?"), exc)
                await turn_context.send_activity(f"Failed to download {getattr(att, 'name', 'file')}: {exc}")

        if prepared:
            await self.orchestrator.process_attachments(
                prepared, context, turn_context, correlation_id, auth_token,
            )

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
        """Proxy a text query to the main app and send the response."""
        started = time.monotonic()

        await turn_context.send_activities([Activity(type=ActivityTypes.typing)])

        req = QueryRequest(
            query=query,
            user_id=context.user_id,
            subscription_id=context.subscription_id,
            tenant_id=tenant_id or "",
            session_id=context.session_id,
        )

        result = await self.query_proxy.ask(req)
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
