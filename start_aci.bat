@echo off
REM ACI System Startup Script for Windows
REM This script sets up the persistent memory infrastructure and starts the ACI system

echo 🚀 Starting ACI System with Persistent Memory...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Create logs directory
if not exist logs mkdir logs

REM Start Docker containers
echo 🐳 Starting PostgreSQL and Redis containers...
docker-compose up -d

REM Wait for PostgreSQL to be ready
echo ⏳ Waiting for PostgreSQL to be ready...
timeout /t 10 /nobreak >nul

REM Check if PostgreSQL is ready
docker-compose exec postgres pg_isready -U aci_user -d aci_memory >nul 2>&1
if errorlevel 1 (
    echo ❌ PostgreSQL is not ready. Waiting longer...
    timeout /t 20 /nobreak >nul
)

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Set environment variables
set DB_HOST=localhost
set DB_PORT=5433
set DB_NAME=aci_memory
set DB_USER=aci_user
set DB_PASSWORD=aci_password
set REDIS_HOST=localhost
set REDIS_PORT=6379

REM Run database migrations/initialization
echo 🗄️  Initializing database schema...
python -c "
from src.memory.persistent_memory import get_memory_manager
memory_manager = get_memory_manager()
print('✅ Database connection established')
"

REM Seed initial memory if needed
echo 🌱 Seeding initial memory data...
python scripts\seed_memory.py

echo 🎉 ACI System is ready!
echo.
echo Available commands:
echo   python scripts\SeedExperience.py    - Run the main simulation
echo   docker-compose logs -f             - View container logs
echo   docker-compose down                - Stop containers
echo.
echo Log files are available in the 'logs/' directory with different resolutions:
echo   logs\level1\  - Top-level thoughts, actions, world events
echo   logs\level2\  - Component interactions, memory operations
echo   logs\level3\  - Detailed internal operations, API calls
echo   logs\level4\  - Full debug information

pause
