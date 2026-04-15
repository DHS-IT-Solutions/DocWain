import re
import httpx

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

EXTRACT_SYSTEM_PROMPT = """/no_think
You are DocWain, a document extraction engine. Extract the content from the provided document text and return it in the requested structured format.

Output format: {output_format}

Rules:
- For "json": Return a JSON object with keys like "document_type", "entities", "tables", "sections", "key_values" as appropriate.
- For "csv": Return CSV-formatted rows with headers.
- For "sections": Return the document broken into labeled sections with their content.
- For "flatfile": Return a flat key-value representation, one per line.
- For "tables": Return all tabular data as arrays of rows.
- Identify entities, classify the document type, and summarize sections alongside the structural output.
- Be thorough and precise. Include all content from the document."""

ANALYZE_SYSTEM_PROMPT = """/no_think
You are DocWain, a document intelligence engine. Analyze the provided document and produce high-level insights.

Analysis type: {analysis_type}

Rules:
- For "summary": Provide a comprehensive summary of the document's content, purpose, and key points.
- For "key_facts": Extract all important facts, figures, dates, names, and data points.
- For "risk_assessment": Identify risks, concerns, compliance issues, and potential problems.
- For "recommendations": Provide actionable recommendations based on the document content.
- For "auto": Choose the most appropriate analysis based on the document type and content.
- Ground every insight in the actual document content. Cite specific sections or quotes as evidence.
- Structure your response as JSON with keys: "summary", "findings" (array), "evidence" (array)."""


class VLLMClient:
    def __init__(self, base_url: str, model: str, timeout: int):
        self._base_url = base_url
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def extract(self, text: str, output_format: str, prompt: str | None) -> str:
        system = EXTRACT_SYSTEM_PROMPT.format(output_format=output_format)
        user_msg = self._build_user_message(text, prompt)
        return await self._call(system, user_msg)

    async def analyze(self, text: str, analysis_type: str, prompt: str | None) -> str:
        system = ANALYZE_SYSTEM_PROMPT.format(analysis_type=analysis_type)
        user_msg = self._build_user_message(text, prompt)
        return await self._call(system, user_msg)

    async def health_check(self) -> bool:
        try:
            health_url = self._base_url.rsplit("/v1", 1)[0] + "/health"
            resp = await self._client.get(health_url)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @staticmethod
    def _build_user_message(text: str, prompt: str | None) -> str | list:
        parts = []
        if prompt:
            parts.append(f"Additional instructions: {prompt}\n\n")

        if text.startswith("data:image/"):
            # Multimodal: image as base64 data URI
            if parts:
                parts.append("Analyze this image:")
            content = []
            if parts:
                content.append({"type": "text", "text": "".join(parts)})
            content.append({"type": "image_url", "image_url": {"url": text}})
            return content

        parts.append(f"Document content:\n{text}")
        return "".join(parts)

    async def _call(self, system: str, user: str | list) -> str:
        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _THINK_RE.sub("", content).strip()

    async def close(self):
        await self._client.aclose()
