# Configuración de Logfire - Paso a Paso

## Problema Actual

El dashboard de Logfire muestra "No data" porque **falta el archivo `.env` con el token de Logfire**.

Aunque la instrumentación está correctamente implementada en el código, Logfire necesita el token para autenticarse y enviar datos al dashboard.

## Solución

### 1. Crear el archivo `.env`

```bash
cp .env.example .env
```

### 2. Obtener tu token de Logfire

1. Ve a https://logfire.pydantic.dev
2. Inicia sesión o crea una cuenta (gratis)
3. Ve a Settings → Tokens
4. Copia tu token

### 3. Editar el archivo `.env`

Abre `.env` y agrega tu token:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=tu_api_key_de_openai

# Logfire Configuration (REQUERIDO para ver datos en el dashboard)
LOGFIRE_TOKEN=tu_token_de_logfire_aqui
```

### 4. Reiniciar el API

```bash
# Detener el API si está corriendo (Ctrl+C)

# Iniciar el API
python -m uvicorn api.main:app --reload
```

Deberías ver en la consola:
```
[OK] Logfire configured successfully
[OK] Logfire FastAPI instrumentation enabled
```

### 5. Ejecutar un job de prueba

Via UI:
```bash
streamlit run ui/app.py
# Subir un CSV y crear un job
```

Via API:
```bash
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -F "dataset=@data/customer_churn.csv" \
  -F "target_column=churn"
```

Via CLI (también funciona):
```bash
python -m automl_agent --data data/customer_churn.csv --target churn
```

### 6. Verificar en Logfire

1. Ve a https://logfire.pydantic.dev
2. Selecciona tu proyecto
3. Ve a "Dashboards" → "LLM Tokens and Costs"
4. Deberías ver datos de tokens y costos

**Nota**: Los datos aparecen en 10-30 segundos después de la llamada a OpenAI.

## Verificación Rápida

Ejecuta este comando para verificar que el token está configurado:

```bash
cat .env | grep LOGFIRE_TOKEN
```

Debería mostrar:
```
LOGFIRE_TOKEN=tu_token_aqui
```

## Troubleshooting

### "LOGFIRE_TOKEN not found"

El archivo `.env` no existe o no tiene el token. Sigue los pasos 1-3 arriba.

### "Logfire not configured"

El token es inválido o ha expirado. Genera un nuevo token en https://logfire.pydantic.dev.

### "No data" en el dashboard después de ejecutar un job

1. Verifica que el job realmente llamó a OpenAI (check logs: `grep "OpenAI" outputs/*/run_summary.json`)
2. Verifica el time range en el dashboard (debe incluir la hora del job)
3. Espera 30 segundos y refresca la página
4. Verifica que viste `[OK] Logfire instrumentation enabled for OpenAI client` en los logs

### Jobs antiguos no muestran datos

Correcto. Solo los jobs ejecutados **después** de configurar Logfire y la instrumentación mostrarán datos.

Los jobs anteriores (antes de Dec 29 19:17) no tienen instrumentación.

## Alternativa: Variables de Entorno

Si prefieres no usar archivo `.env`, puedes exportar las variables:

```bash
export OPENAI_API_KEY="tu_api_key"
export LOGFIRE_TOKEN="tu_token"
python -m uvicorn api.main:app --reload
```

**Importante**: Necesitas exportarlas en cada sesión o agregarlas a tu `.bashrc`/`.zshrc`.
