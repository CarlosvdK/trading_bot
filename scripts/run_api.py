#!/usr/bin/env python3
"""Start the FastAPI backend."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
