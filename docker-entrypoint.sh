#!/bin/bash
set -e

echo "============================================"
echo "  AutoML Agent - Starting services..."
echo "============================================"

# Auto-detect LLM provider: Azure > OpenAI > Ollama
# Check for valid API keys (not placeholder values)
AZURE_VALID=""
OPENAI_VALID=""

if [ -n "$AZURE_OPENAI_API_KEY" ] && [ "$AZURE_OPENAI_API_KEY" != "your_azure_api_key_here" ] && [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
    AZURE_VALID="true"
fi

if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "your_openai_api_key_here" ]; then
    OPENAI_VALID="true"
fi

# Determine provider
if [ -n "$AZURE_VALID" ]; then
    echo "[AUTO-DETECT] Found Azure OpenAI credentials"
    echo "[INFO] Using Azure OpenAI as LLM provider"
    LLM_PROVIDER="azure"
elif [ -n "$OPENAI_VALID" ]; then
    echo "[AUTO-DETECT] Found OpenAI API key"
    echo "[INFO] Using OpenAI as LLM provider"
    LLM_PROVIDER="openai"
else
    echo "[AUTO-DETECT] No valid API keys found"
    echo "[INFO] Using Ollama (local) as LLM provider"
    LLM_PROVIDER="ollama"

    # Wait for Ollama to be ready
    echo "[INFO] Waiting for Ollama to be ready at $OLLAMA_HOST..."
    max_attempts=60
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
            echo "[OK] Ollama is ready!"
            break
        fi
        attempt=$((attempt + 1))
        echo "[...] Waiting for Ollama (attempt $attempt/$max_attempts)..."
        sleep 2
    done

    if [ $attempt -eq $max_attempts ]; then
        echo "[WARN] Ollama not responding, continuing anyway..."
    fi
fi

# Update config to use environment-provided Ollama host (for fallback)
if [ -n "$OLLAMA_HOST" ]; then
    sed -i "s|ollama_host:.*|ollama_host: \"$OLLAMA_HOST\"|g" automl_agent/config.yaml
fi

# Start FastAPI in background
echo "[INFO] Starting FastAPI server on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
sleep 3

# Start Streamlit
echo "[INFO] Starting Streamlit UI on port 8501..."
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
UI_PID=$!

echo "============================================"
echo "  AutoML Agent is running!"
echo "  - API:  http://localhost:8000"
echo "  - UI:   http://localhost:8501"
echo "  - Docs: http://localhost:8000/docs"
echo "============================================"

# Wait for either process to exit
wait -n $API_PID $UI_PID

# If one exits, kill the other and exit
kill $API_PID $UI_PID 2>/dev/null
exit 1
