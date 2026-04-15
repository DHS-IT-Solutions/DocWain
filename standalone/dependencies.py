from pymongo import MongoClient

from standalone.config import Config
from standalone.vllm_client import VLLMClient

_mongo_client: MongoClient | None = None
_vllm_client: VLLMClient | None = None


def get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(Config.MONGODB_URI)
    return _mongo_client[Config.MONGODB_DB]


def get_vllm_client() -> VLLMClient:
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = VLLMClient(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL_NAME,
            timeout=Config.VLLM_TIMEOUT,
        )
    return _vllm_client


async def cleanup():
    if _vllm_client:
        await _vllm_client.close()
    if _mongo_client:
        _mongo_client.close()
