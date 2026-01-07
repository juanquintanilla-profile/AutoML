# AutoML Agent - Start script with auto-browser open
# Usage: .\start.ps1

Write-Host "Starting AutoML services..." -ForegroundColor Cyan
docker-compose up -d

Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
$maxAttempts = 60
$attempt = 0

while ($attempt -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            break
        }
    } catch {
        $attempt++
        Write-Host "  Waiting... ($attempt/$maxAttempts)" -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
}

if ($attempt -ge $maxAttempts) {
    Write-Host "Timeout waiting for services. Check docker-compose logs." -ForegroundColor Red
    exit 1
}

Write-Host "`nServices are ready!" -ForegroundColor Green
Write-Host "Opening browser..." -ForegroundColor Cyan
Start-Process "http://localhost:8501"

Write-Host "`nAutoML is running:" -ForegroundColor Green
Write-Host "  - UI:  http://localhost:8501" -ForegroundColor White
Write-Host "  - API: http://localhost:8000/docs" -ForegroundColor White
Write-Host "`nTo view logs: docker-compose logs -f" -ForegroundColor Gray
Write-Host "To stop: docker-compose down" -ForegroundColor Gray
