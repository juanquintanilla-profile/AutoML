# AutoML Agent

Sistema multi-agente de AutoML que utiliza orquestación basada en LLM para automatizar flujos de trabajo de machine learning.

## Cómo Ejecutar la Aplicación

### Requisitos Previos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución

**NOTA IMPORTANTE**: No necesitas configurar ningún archivo `.env` ni claves API para empezar. El sistema usa **Ollama** (LLM local gratuito) por defecto. Solo necesitas el archivo `.env` si quieres usar OpenAI o Azure OpenAI.

### Paso 1: Generar y Ejecutar los Contenedores

**Windows (CMD):**
```cmd
start.bat
```

**Windows (PowerShell):**
```powershell
.\start.ps1
```

**Linux/Mac:**
```bash
docker-compose up -d
```

### Paso 2: Esperar la Inicialización

En la primera ejecución, el sistema:
1. Construye los contenedores de Docker
2. Descarga el modelo LLM (~4GB) - esto puede tomar varios minutos
3. Inicia todos los servicios

Puedes ver el progreso con:
```bash
docker-compose logs -f
```

### Paso 3: Acceder a la Aplicación

| Servicio | URL |
|----------|-----|
| Interfaz Web (Streamlit) | http://localhost:8501 |
| API (Swagger) | http://localhost:8000/docs |

### Comandos Útiles

```bash
docker-compose up       # Iniciar todos los servicios
docker-compose up -d    # Iniciar en segundo plano
docker-compose down     # Detener todos los servicios
docker-compose logs -f  # Ver logs en tiempo real
docker-compose ps       # Ver estado de los contenedores
```

---

## Arquitectura

```
automl_agent/
│
├── main.py                    # Punto de entrada
├── config.yaml                # Configuración (budget, métricas, límites)
│
├── orchestrator/
│   ├── planner.py             # Decisor basado en LLM
│   └── state.py               # Estado global del sistema
│
├── agents/
│   ├── data_agent.py          # Análisis de datos y preprocesamiento
│   ├── modeling_agent.py      # Generación de pipelines
│   ├── hpo_agent.py           # Optimización de hiperparámetros (FLAML/Optuna)
│   └── eval_agent.py          # Evaluación y comparación de modelos
│
├── tools/
│   ├── data_utils.py          # Carga, división y validación de datos
│   ├── preprocessing.py       # Encoders, escalado, imputación
│   ├── metrics.py             # Métricas de evaluación
│   └── logging.py             # Integración con MLflow
│
├── memory/
│   └── runs.json              # Historial de ejecuciones
│
├── prompts/
│   └── planner.txt            # Prompt del orquestador LLM
│
└── output/
    ├── model.joblib           # Modelo final
    ├── metrics.json           # Resultados
    └── run_summary.json       # Traza de ejecución
```

## Stack Tecnológico

### LLM / Planner
- `ollama` - LLM local (predeterminado, gratuito, sin API key)
- `openai` - GPT-4 para decisiones de orquestación
- `azure` - Azure OpenAI para despliegues empresariales

### AutoML / Búsqueda
- `flaml` - Optimización AutoML rápida
- `optuna` - Ajuste de hiperparámetros

### Modelado
- `scikit-learn` - Pipelines de ML
- `lightgbm` - Gradient boosting
- `xgboost` - Gradient boosting
- `catboost` - Gradient boosting

### Datos
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `ydata-profiling` - Perfilado de datos (opcional)

### Evaluación
- `scikit-learn` - Métricas
- `mlflow` - Tracking de experimentos

### Infraestructura
- `joblib` - Serialización de modelos
- `pyyaml` - Configuración

## Instalación Local (Alternativa)

Si prefieres ejecutar sin Docker:

### Opción 1: Con Ollama (Gratuito, Sin API Key)

```bash
# 1. Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh   # Linux/Mac
# Windows: Descargar desde https://ollama.ai

# 2. Descargar el modelo
ollama pull llama3.2

# 3. Instalar dependencias de Python
pip install -r requirements.txt
pip install -r requirements-api.txt

# 4. Ejecutar
python -m automl_agent --data data/customer_churn.csv --target churn
```

### Opción 2: Con OpenAI

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar entorno
cp .env.example .env
# Editar .env y agregar tu OPENAI_API_KEY

# 3. Actualizar config.yaml
# Establecer llm.provider: "openai"
```

### Opción 3: Con Azure OpenAI

```bash
cp .env.example .env
# Editar .env y agregar:
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_KEY
```

## Configuración para Azure OpenAI

Si estás usando **Azure OpenAI** (ej. GPT-4o-mini desde Azure ML), sigue estos pasos:

### 1. Actualizar `config.yaml`:

```yaml
llm:
  provider: "azure"
  deployment_name: "gpt-4o-mini"  # Tu nombre de deployment en Azure
  api_version: "2024-02-15-preview"
  temperature: 0.7
  max_tokens: 2000
```

O usa el ejemplo proporcionado:
```bash
cp automl_agent/config.azure.yaml automl_agent/config.yaml
```

### 2. Establecer variables de entorno en `.env`:

```bash
AZURE_OPENAI_ENDPOINT=https://tu-recurso.openai.azure.com/
AZURE_OPENAI_API_KEY=tu_azure_openai_api_key_aqui
```

### 3. Ejecutar normalmente:

```bash
python -m automl_agent --data data.csv --target columna_objetivo
```

## Selección Automática de Proveedor LLM

El sistema **auto-detecta** el proveedor LLM disponible basándose en las variables de entorno, siguiendo esta prioridad:

1. **Azure OpenAI** (si existen `AZURE_OPENAI_ENDPOINT` y `AZURE_OPENAI_API_KEY`)
2. **OpenAI** (si existe `OPENAI_API_KEY`)
3. **Ollama** (fallback por defecto - **no requiere API keys**)

Esto significa:
- **Sin archivo .env**: usa Ollama (local, gratuito)
- **Con .env vacío**: usa Ollama (local, gratuito)
- **Con claves Azure en .env**: usa Azure OpenAI automáticamente
- **Con clave OpenAI en .env**: usa OpenAI automáticamente

No necesitas modificar `config.yaml` para cambiar de proveedor, el sistema lo detecta automáticamente al iniciar.

## Prueba Rápida

### 1. Generar Dataset de Ejemplo

```bash
python generate_example_data.py
```

Esto crea `data/customer_churn.csv` con:
- 1000 muestras
- 10 features (numéricos + categóricos)
- Tarea de clasificación binaria
- Algunos valores faltantes para probar preprocesamiento

### 2. Ejecutar Script de Prueba

```bash
# Asegúrate de que el venv esté activado
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Ejecutar la prueba
./test_automl.sh
```

O manualmente:

```bash
python -m automl_agent \
  --data data/customer_churn.csv \
  --target churn
```

### 3. Revisar Resultados

Después de completar la ejecución, revisa:
- `automl_agent/output/model.joblib` - Mejor modelo entrenado
- `automl_agent/output/metrics.json` - Métricas de rendimiento
- `automl_agent/output/run_summary.json` - Traza completa de ejecución

## Uso

### Uso Básico

```bash
python -m automl_agent \
  --data ruta/a/datos.csv \
  --target nombre_columna_objetivo
```

### Con Configuración Personalizada

```bash
python -m automl_agent \
  --data ruta/a/datos.csv \
  --target nombre_columna_objetivo \
  --config ruta/a/config_personalizado.yaml \
  --output ruta/a/directorio_salida
```

## Configuración

Edita `automl_agent/config.yaml` para personalizar:

- **Budget**: Máximo de iteraciones, límites de tiempo, early stopping
- **LLM**: Selección de modelo, temperatura, max tokens
- **Metrics**: Métricas primarias y secundarias
- **Model Families**: Qué algoritmos probar
- **HPO**: Motor de optimización y parámetros
- **Data**: División train/test, opciones de preprocesamiento
- **Logging**: Configuración de MLflow

## Cómo Funciona

### Flujo de Trabajo

```
main.py
  ↓
Data Agent → Analizar dataset, proponer preprocesamiento
  ↓
Planner (LLM) → Decidir siguiente acción
  ↓
Modeling Agent → Generar candidatos de modelos
  ↓
HPO Agent → Optimizar hiperparámetros (FLAML/Optuna)
  ↓
Evaluation Agent → Comparar modelos
  ↓
Planner → Decidir: iterar o detener
```

### Comunicación entre Agentes

Los agentes se comunican mediante **estado estructurado** (JSON), no lenguaje natural:

```json
{
  "action": "run_hpo",
  "parameters": {"model_family": "lightgbm"},
  "reason": "mejor balance hasta ahora"
}
```

### Qué Hace el LLM

✅ Decide qué agente ejecutar a continuación
✅ Prioriza qué modelos optimizar
✅ Decide cuándo detenerse

❌ NO entrena modelos
❌ NO ve los datos crudos
❌ NO calcula métricas

### Condiciones de Parada

- Budget agotado (tiempo o iteraciones)
- Sin mejora durante N iteraciones
- Score objetivo alcanzado

## Artefactos de Salida

Después de completar, revisa `automl_agent/output/`:

- `model.joblib` - Pipeline del mejor modelo
- `metrics.json` - Resultados de evaluación
- `run_summary.json` - Traza completa de ejecución
- `config.json` - Configuración utilizada

## Ejemplo de Uso Programático

```python
from automl_agent.main import run_automl

state = run_automl(
    data_path="data/train.csv",
    target_column="target",
    config_path="automl_agent/config.yaml",
    output_dir="automl_agent/output"
)

print(f"Mejor modelo: {state.best_model['model_name']}")
print(f"Mejor score: {state.best_score}")
```

## Extender el Sistema

### Agregar un Nuevo Agente

1. Crear `agents/nuevo_agente.py`
2. Implementar método `execute()`
3. Actualizar `planner.txt` para describir cuándo usarlo
4. Agregar manejador de acción en `main.py`

### Agregar Métricas Personalizadas

1. Editar `tools/metrics.py`
2. Agregar a `CLASSIFICATION_METRICS` o `REGRESSION_METRICS`
3. Actualizar `config.yaml` para usarla

### Cambiar Motor de HPO

En `config.yaml`:
```yaml
hpo:
  engine: "optuna"  # o "flaml"
```

## Licencia

MIT
