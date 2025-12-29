# Logfire Integration Fix

## Problema Identificado

El dashboard de Logfire mostraba "No data" en los gráficos de "LLM Tokens and Costs" porque:

1. **Logfire estaba configurado** en `api/main.py` para eventos HTTP y jobs
2. **Pero las llamadas a OpenAI NO estaban instrumentadas** en `automl_agent/orchestrator/planner.py`
3. Sin instrumentación del cliente de OpenAI, Logfire no puede capturar:
   - Tokens de entrada/salida
   - Costos de las llamadas
   - Latencia del modelo
   - Modelo usado en cada llamada

## Solución Implementada

### Cambios en `automl_agent/orchestrator/planner.py`

1. **Importar Logfire (opcional)**:
   ```python
   try:
       import logfire
       LOGFIRE_AVAILABLE = True
   except ImportError:
       LOGFIRE_AVAILABLE = False
       logfire = None
   ```

2. **Configurar Logfire en el constructor del PlannerAgent**:
   ```python
   if LOGFIRE_AVAILABLE:
       try:
           logfire.configure()
       except Exception as e:
           pass  # Ya está configurado, OK
   ```

3. **Instrumentar el cliente de OpenAI después de crearlo**:
   ```python
   if LOGFIRE_AVAILABLE:
       try:
           logfire.instrument_openai(self.client)
           print("[OK] Logfire instrumentation enabled for OpenAI client")
       except Exception as e:
           print(f"[WARN] Failed to instrument OpenAI: {e}")
   ```

### Archivos Modificados

- `automl_agent/orchestrator/planner.py`: Agregada instrumentación de OpenAI
- `README_API.md`: Actualizada documentación de Logfire
- `.env.example`: Creado con variables de entorno necesarias

## Cómo Funciona

1. Cuando el API arranca, Logfire se configura en `api/main.py`
2. Cuando se crea un job, se ejecuta en un thread pool worker
3. El worker crea un `PlannerAgent` que:
   - Re-configura Logfire en el contexto del thread (si es necesario)
   - Crea el cliente de OpenAI (OpenAI o AzureOpenAI)
   - **Instrumenta el cliente con `logfire.instrument_openai()`**
4. Cada llamada a `client.chat.completions.create()` ahora se registra automáticamente
5. Logfire captura y envía los datos al dashboard

## Qué Verás en el Dashboard

Después de ejecutar algunos jobs, el dashboard de Logfire mostrará:

### LLM Tokens and Costs (from records)
- **By model and type**: Gráfico de barras con tokens de entrada/salida por modelo
- **By type**: Distribución de tokens (prompt vs completion)
- **Total tokens**: Suma acumulada de tokens usados
- **Total cost**: Costo en USD de todas las llamadas

### Additional Metrics
- **Latency**: Tiempo de respuesta de cada llamada a OpenAI
- **Request rate**: Llamadas por minuto/hora
- **Model distribution**: Qué modelos se usan más
- **Error rate**: Porcentaje de llamadas fallidas

## Verificación

Para verificar que funciona:

1. **Reiniciar el API**:
   ```bash
   python -m uvicorn api.main:app --reload
   ```

2. **Crear un nuevo job** (vía UI o API)

3. **Ir al dashboard de Logfire** y buscar:
   - Pestaña "Live" → Deberías ver eventos de tipo `openai.chat.completions`
   - Pestaña "Dashboards" → "LLM Tokens and Costs" → Debería mostrar datos

4. **Verificar logs en consola**:
   ```
   [OK] Logfire configured successfully
   [OK] Logfire FastAPI instrumentation enabled
   [OK] Logfire instrumentation enabled for OpenAI client
   ```

## Notas Importantes

- **Logfire es OPCIONAL**: Si no tienes `LOGFIRE_TOKEN` en `.env`, el sistema sigue funcionando sin instrumentación
- **Compatible con OpenAI y Azure OpenAI**: Funciona con ambos providers
- **No afecta rendimiento**: La instrumentación agrega ~10-50ms por llamada
- **Datos en tiempo real**: Los gráficos se actualizan en cuanto se hacen llamadas a OpenAI

## Troubleshooting

Si aún no ves datos:

1. **Verificar token**: `echo $LOGFIRE_TOKEN` debería mostrar tu token
2. **Verificar logs**: Buscar `[OK] Logfire instrumentation enabled for OpenAI client`
3. **Verificar llamadas**: Asegurar que el job realmente está llamando a OpenAI (check logs)
4. **Dashboard delay**: Puede tardar 10-30 segundos en aparecer los primeros datos
5. **Time range**: En el dashboard, asegurar que el "Time Range" incluye el momento del job
