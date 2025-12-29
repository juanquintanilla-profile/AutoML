#!/bin/bash
# Script para probar Logfire con un job rápido

echo "=== Test de Logfire ==="
echo ""
echo "Este script va a:"
echo "1. Ejecutar un job de prueba con el dataset de ejemplo"
echo "2. La instrumentación de Logfire capturará los tokens"
echo "3. Verás los datos en el dashboard en ~30 segundos"
echo ""
echo "Dashboard: https://logfire.pydantic.dev"
echo ""
read -p "Presiona Enter para continuar..."

echo ""
echo "Ejecutando AutoML job..."
echo ""

python -m automl_agent \
  --data data/customer_churn.csv \
  --target churn \
  --output automl_agent/output

echo ""
echo "=== Job completado ==="
echo ""
echo "Ahora:"
echo "1. Ve a https://logfire.pydantic.dev"
echo "2. Selecciona tu proyecto"
echo "3. Ve a 'Dashboards' → 'LLM Tokens and Costs'"
echo "4. Deberías ver datos de tokens y costos"
echo ""
echo "Si no ves datos:"
echo "- Espera 30 segundos y refresca"
echo "- Verifica el time range del dashboard"
echo "- Ejecuta: ./check_logfire_setup.sh para diagnosticar"
echo ""
