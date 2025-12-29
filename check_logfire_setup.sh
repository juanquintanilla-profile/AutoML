#!/bin/bash
# Script para verificar la configuración de Logfire

echo "=== Verificación de Configuración de Logfire ==="
echo ""

# Check 1: .env file
echo "1. Verificando archivo .env..."
if [ -f ".env" ]; then
    echo "   ✓ Archivo .env encontrado"

    # Check for LOGFIRE_TOKEN
    if grep -q "^LOGFIRE_TOKEN=" .env; then
        TOKEN=$(grep "^LOGFIRE_TOKEN=" .env | cut -d'=' -f2)
        if [ -z "$TOKEN" ] || [ "$TOKEN" = "your_logfire_token_here" ]; then
            echo "   ✗ LOGFIRE_TOKEN está vacío o es el valor por defecto"
            echo "     → Edita .env y agrega tu token de https://logfire.pydantic.dev"
        else
            echo "   ✓ LOGFIRE_TOKEN configurado (${TOKEN:0:10}...)"
        fi
    else
        echo "   ✗ LOGFIRE_TOKEN no encontrado en .env"
        echo "     → Agrega: LOGFIRE_TOKEN=tu_token_aqui"
    fi

    # Check for OPENAI_API_KEY
    if grep -q "^OPENAI_API_KEY=" .env; then
        KEY=$(grep "^OPENAI_API_KEY=" .env | cut -d'=' -f2)
        if [ -z "$KEY" ] || [ "$KEY" = "your_openai_api_key_here" ]; then
            echo "   ✗ OPENAI_API_KEY está vacío o es el valor por defecto"
        else
            echo "   ✓ OPENAI_API_KEY configurado (${KEY:0:10}...)"
        fi
    else
        echo "   ⚠ OPENAI_API_KEY no encontrado (puede estar usando Azure)"
    fi
else
    echo "   ✗ Archivo .env NO encontrado"
    echo "     → Ejecuta: cp .env.example .env"
    echo "     → Luego edita .env con tus tokens"
fi

echo ""

# Check 2: Logfire package
echo "2. Verificando paquete logfire..."
if python -c "import logfire" 2>/dev/null; then
    VERSION=$(python -c "import logfire; print(logfire.__version__)" 2>/dev/null)
    echo "   ✓ logfire instalado (version $VERSION)"
else
    echo "   ✗ logfire NO instalado"
    echo "     → Ejecuta: pip install -r requirements-api.txt"
fi

echo ""

# Check 3: OpenAI package
echo "3. Verificando paquete openai..."
if python -c "import openai" 2>/dev/null; then
    VERSION=$(python -c "import openai; print(openai.__version__)" 2>/dev/null)
    echo "   ✓ openai instalado (version $VERSION)"
else
    echo "   ✗ openai NO instalado"
    echo "     → Ejecuta: pip install -r requirements.txt"
fi

echo ""

# Check 4: Recent jobs
echo "4. Verificando jobs recientes..."
if [ -d "outputs" ]; then
    JOB_COUNT=$(ls -d outputs/*/ 2>/dev/null | wc -l)
    echo "   ℹ Total de jobs: $JOB_COUNT"

    LATEST=$(ls -t outputs/*/run_summary.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        TIMESTAMP=$(stat -c %y "$LATEST" | cut -d' ' -f1,2 | cut -d'.' -f1)
        echo "   ℹ Job más reciente: $TIMESTAMP"

        # Check if job is after instrumentation
        INSTRUMENT_TIME="2025-12-29 19:17:00"
        if [[ "$TIMESTAMP" > "$INSTRUMENT_TIME" ]]; then
            echo "   ✓ Job ejecutado DESPUÉS de la instrumentación"
        else
            echo "   ⚠ Job ejecutado ANTES de la instrumentación (no tendrá datos de Logfire)"
            echo "     → Ejecuta un nuevo job para ver datos"
        fi
    fi
else
    echo "   ℹ No hay jobs ejecutados aún"
fi

echo ""

# Check 5: Instrumentation code
echo "5. Verificando código de instrumentación..."
if grep -q "logfire.instrument_openai" automl_agent/orchestrator/planner.py; then
    echo "   ✓ Instrumentación presente en planner.py"
else
    echo "   ✗ Instrumentación NO encontrada"
    echo "     → Código desactualizado, ejecuta: git pull"
fi

echo ""
echo "=== Resumen ==="
echo ""
echo "Para que Logfire funcione necesitas:"
echo "1. ✓ Archivo .env con LOGFIRE_TOKEN válido"
echo "2. ✓ Paquetes logfire y openai instalados"
echo "3. ✓ Ejecutar un job NUEVO después de configurar todo"
echo "4. ✓ Esperar 10-30 segundos y verificar el dashboard"
echo ""
echo "Dashboard: https://logfire.pydantic.dev"
echo "Docs: Ver SETUP_LOGFIRE.md para instrucciones completas"
echo ""
