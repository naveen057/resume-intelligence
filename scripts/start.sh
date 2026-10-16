#!/usr/bin/env bash
# One container, two processes: FastAPI on :8000 (internal) and Streamlit on $PORT (public).
set -euo pipefail
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
export API_URL="http://localhost:8000"
exec streamlit run ui/streamlit_app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0 \
     --server.headless true --browser.gatherUsageStats false
