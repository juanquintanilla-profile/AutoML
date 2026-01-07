# AutoML Agent API + UI

Extensión del AutoML Agent con una API REST y una interfaz web, **sin modificar el código original de `automl_agent/`**.

## Arquitectura

```
┌──────────────┐
│ Streamlit UI │  (localhost:8501)
└──────┬───────┘
       │ HTTP
┌──────▼───────┐
│  FastAPI     │  (localhost:8000)
│  + Logfire   │
└──────┬───────┘
       │ Llamadas Python
┌──────▼───────┐
│ automl_agent │  (sin cambios)
└──────────────┘
```

## Inicio Rápido

### 1. Instalar Dependencias

```bash
# Instalar dependencias base de AutoML
pip install -r requirements.txt

# Instalar dependencias de API + UI
pip install -r requirements-api.txt
```

### 2. Configurar Logfire (Opcional)

```bash
# Crear archivo .env
cp .env.example .env

# Obtener token gratuito en https://logfire.pydantic.dev
# Agregar a .env:
# LOGFIRE_TOKEN=tu_token_aqui
```

### 3. Iniciar la API

```bash
# Terminal 1: Iniciar servidor FastAPI
python -m uvicorn api.main:app --reload

# API disponible en http://localhost:8000
# Documentación Swagger en http://localhost:8000/docs
```

### 4. Iniciar la UI

```bash
# Terminal 2: Iniciar Streamlit
streamlit run ui/app.py

# UI se abrirá en http://localhost:8501
```

## Uso

### Vía Interfaz Web (Streamlit)

1. Ir a http://localhost:8501
2. Subir un dataset CSV
3. Seleccionar columna objetivo
4. Hacer clic en "Iniciar Trabajo AutoML"
5. Monitorear progreso en la pestaña "Monitor de Trabajos"
6. Ver resultados en la pestaña "Resultados"

### Vía API (curl)

```bash
# Crear trabajo
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -F "dataset=@data/customer_churn.csv" \
  -F "target_column=churn"

# Respuesta: {"job_id": "abc12345", "status": "pending", ...}

# Verificar estado
curl "http://localhost:8000/api/v1/jobs/abc12345/status"

# Obtener resultados (cuando esté completado)
curl "http://localhost:8000/api/v1/jobs/abc12345/results"

# Descargar modelo
curl "http://localhost:8000/api/v1/jobs/abc12345/download-model" -O

# Listar todos los trabajos
curl "http://localhost:8000/api/v1/jobs"
```

### Vía Python

```python
import requests

# Crear trabajo
with open('data/customer_churn.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/jobs',
        files={'dataset': f},
        data={'target_column': 'churn'}
    )

job_id = response.json()['job_id']

# Consultar estado
import time
while True:
    status = requests.get(f'http://localhost:8000/api/v1/jobs/{job_id}/status').json()
    print(f"Estado: {status['status']}")

    if status['status'] in ['completed', 'failed']:
        break

    time.sleep(5)

# Obtener resultados
results = requests.get(f'http://localhost:8000/api/v1/jobs/{job_id}/results').json()
print(f"Mejor modelo: {results['best_model']}")
print(f"Mejor score: {results['best_score']}")
```

## Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Verificación de salud |
| `POST` | `/api/v1/jobs` | Crear nuevo trabajo AutoML |
| `GET` | `/api/v1/jobs` | Listar todos los trabajos |
| `GET` | `/api/v1/jobs/{id}/status` | Obtener estado del trabajo |
| `GET` | `/api/v1/jobs/{id}/results` | Obtener resultados del trabajo |
| `GET` | `/api/v1/jobs/{id}/logs` | Obtener logs del trabajo |
| `GET` | `/api/v1/jobs/{id}/download-model` | Descargar modelo entrenado |

Documentación completa de la API: http://localhost:8000/docs

## Monitoreo con Logfire

Todas las solicitudes a la API, llamadas al LLM y trabajos de AutoML se rastrean automáticamente con Logfire:

1. Registrarse en https://logfire.pydantic.dev (nivel gratuito disponible)
2. Obtener tu token y agregarlo al archivo `.env`:
   ```bash
   LOGFIRE_TOKEN=tu_token_aqui
   ```
3. Iniciar la API (Logfire se configurará automáticamente)
4. Ejecutar algunos trabajos de AutoML
5. Ver trazas en tiempo real en el dashboard de Logfire

Podrás ver:
- **Tokens y Costos del LLM**: Uso de tokens, costos del modelo y latencia para todas las llamadas a OpenAI/Azure
- **Solicitudes API**: Todas las solicitudes/respuestas HTTP con tiempos
- **Ejecución de Trabajos**: Líneas de tiempo completas de trabajos AutoML con transiciones de agentes
- **Errores**: Stack traces completos con contexto
- **Rendimiento**: Tiempos de respuesta, throughput y cuellos de botella

El PlannerAgent instrumenta automáticamente todas las llamadas a la API de OpenAI, por lo que verás gráficos detallados de uso de tokens en el dashboard "LLM Tokens and Costs".

## Salidas

Los artefactos de los trabajos se guardan en `outputs/{job_id}/`:
- `model.joblib` - Modelo entrenado
- `metrics.json` - Métricas de evaluación
- `run_summary.json` - Resumen completo de ejecución
- `config.json` - Configuración utilizada

La base de datos SQLite está en `outputs/jobs.db`.

## ¿Qué Cambió?

**¡Nada en `automl_agent/`!** El código original está completamente intacto.

Archivos nuevos:
- `api/` - Aplicación FastAPI
- `ui/` - Aplicación Streamlit
- `requirements-api.txt` - Dependencias adicionales
- `outputs/` - Salidas de trabajos y base de datos

## Despliegue en Producción

Para producción, considerar:

1. **Variables de entorno**: Usar gestión adecuada de secretos
2. **Base de datos**: Cambiar de SQLite a PostgreSQL
3. **Almacenamiento de archivos**: Usar S3 o similar para datasets/modelos
4. **Cola**: Agregar Celery + Redis para cola de trabajos robusta
5. **Autenticación**: Agregar autenticación/autorización
6. **CORS**: Restringir a orígenes específicos
7. **HTTPS**: Usar proxy reverso (nginx) con SSL

Ejemplo de Docker Compose:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOGFIRE_TOKEN=${LOGFIRE_TOKEN}

  ui:
    build: .
    command: streamlit run ui/app.py
    ports:
      - "8501:8501"
```

## Solución de Problemas

**La API no inicia:**
- Verificar si el puerto 8000 está disponible
- Verificar que `.env` tenga un `LOGFIRE_TOKEN` válido (o eliminar la variable si no se usa)
- Asegurar que `requirements-api.txt` esté instalado

**Los trabajos fallan:**
- Verificar que `automl_agent/config.yaml` sea válido
- Verificar que el dataset tenga el formato correcto
- Revisar el dashboard de Logfire para errores detallados

**La UI no puede conectarse:**
- Asegurar que la API esté corriendo en el puerto 8000
- Verificar configuración de CORS en `api/main.py`
