import os


class Config:
    PORT = int(os.getenv("STANDALONE_PORT", "8400"))
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8100/v1")
    VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "docwain")
    VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "120"))
    MONGODB_URI = os.getenv("STANDALONE_MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB = os.getenv("STANDALONE_MONGODB_DB", "docwain_standalone")
    ADMIN_SECRET = os.getenv("STANDALONE_ADMIN_SECRET", "")
    MAX_FILE_SIZE_MB = int(os.getenv("STANDALONE_MAX_FILE_SIZE_MB", "50"))
    LOG_LEVEL = os.getenv("STANDALONE_LOG_LEVEL", "INFO")
