import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_extract_builds_correct_request():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"tables": [["A", "B"], [1, 2]]}'}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.extract("Invoice total: $500", output_format="json", prompt=None)

    assert result == '{"tables": [["A", "B"], [1, 2]]}'
    call_args = mock_client.post.call_args
    body = call_args.kwargs["json"]
    assert body["model"] == "docwain-fast"
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert "Invoice total: $500" in body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_extract_includes_user_prompt():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "extracted content"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    await client.extract("doc text", output_format="tables", prompt="focus on financial data")

    body = mock_client.post.call_args.kwargs["json"]
    user_content = body["messages"][1]["content"]
    assert "focus on financial data" in user_content
    assert "doc text" in user_content


@pytest.mark.asyncio
async def test_analyze_builds_correct_request():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Risk assessment: moderate risk found"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.analyze("contract text", analysis_type="risk_assessment", prompt=None)

    assert "Risk assessment" in result
    body = mock_client.post.call_args.kwargs["json"]
    assert body["messages"][0]["role"] == "system"
    assert "risk_assessment" in body["messages"][0]["content"].lower() or "risk" in body["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_analyze_with_custom_prompt():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "analysis result"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    await client.analyze("doc text", analysis_type="summary", prompt="focus on compliance")

    body = mock_client.post.call_args.kwargs["json"]
    user_content = body["messages"][1]["content"]
    assert "focus on compliance" in user_content


@pytest.mark.asyncio
async def test_strips_think_tags():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "<think>reasoning here</think>The actual answer"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.extract("text", output_format="json", prompt=None)
    assert "<think>" not in result
    assert "The actual answer" in result


@pytest.mark.asyncio
async def test_health_check():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_extract_image_sends_multimodal():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"document_type": "receipt"}'}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    image_data_uri = "data:image/png;base64,iVBORw0KGgo="
    await client.extract(image_data_uri, output_format="json", prompt=None)

    body = mock_client.post.call_args.kwargs["json"]
    user_content = body["messages"][1]["content"]
    # Should be a list (multimodal) not a string
    assert isinstance(user_content, list)
    assert any(item["type"] == "image_url" for item in user_content)


@pytest.mark.asyncio
async def test_health_check_failure():
    from standalone.vllm_client import VLLMClient
    import httpx

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.health_check()
    assert result is False
