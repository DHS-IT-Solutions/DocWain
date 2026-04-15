import uvicorn
from standalone.config import Config

if __name__ == "__main__":
    uvicorn.run(
        "standalone.app:app",
        host="0.0.0.0",
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower(),
    )
