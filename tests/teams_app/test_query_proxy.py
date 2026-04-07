import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from teams_app.proxy.query_proxy import QueryProxy, QueryRequest, QueryResult


def test_query_request_builds_payload():
    req = QueryRequest(
        query="what is revenue?",
        user_id="teams_user_123",
        subscription_id="teams_tenant_abc",
    )
    payload = req.to_dict()
    assert payload["query"] == "what is revenue?"
    assert payload["user_id"] == "teams_user_123"
    assert payload["subscription_id"] == "teams_tenant_abc"
    assert payload["stream"] is False
    assert payload["profile_id"] == "default"


def test_query_request_headers():
    req = QueryRequest(
        query="test",
        user_id="u1",
        subscription_id="teams_t1",
        tenant_id="t1",
    )
    headers = req.headers()
    assert headers["x-source"] == "teams"
    assert headers["x-teams-tenant"] == "t1"
    assert headers["Content-Type"] == "application/json"


def test_query_result_from_sse_data():
    data = {
        "response": "Revenue is $1M",
        "sources": [{"title": "report.pdf"}],
        "grounded": True,
        "context_found": True,
    }
    result = QueryResult.from_response(data)
    assert result.response == "Revenue is $1M"
    assert result.context_found is True
    assert len(result.sources) == 1


def test_query_result_no_context():
    data = {"response": "I don't know", "sources": [], "grounded": False, "context_found": False}
    result = QueryResult.from_response(data)
    assert result.context_found is False


@pytest.mark.asyncio
async def test_proxy_returns_result_on_success():
    proxy = QueryProxy(main_app_url="http://localhost:8000", timeout=10)
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "response": "Answer here",
        "sources": [],
        "grounded": True,
        "context_found": True,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client_instance

        req = QueryRequest(query="test", user_id="u1", subscription_id="t1")
        result = await proxy.ask(req)
        assert result.response == "Answer here"
        assert result.context_found is True


@pytest.mark.asyncio
async def test_proxy_returns_error_on_failure():
    proxy = QueryProxy(main_app_url="http://localhost:8000", timeout=10)

    with patch("httpx.AsyncClient") as MockClient:
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = Exception("Connection refused")
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client_instance

        req = QueryRequest(query="test", user_id="u1", subscription_id="t1")
        result = await proxy.ask(req)
        assert result.error is not None
        assert result.context_found is False
