#!/bin/bash
# ACI System Startup Script
# This script sets up the persistent memory infrastructure and starts the ACI system

set -e

echo "🚀 Starting ACI System with Persistent Memory..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start Docker containers
echo "🐳 Starting PostgreSQL and Redis containers..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 10

# Check if PostgreSQL is ready
if ! docker-compose exec -T postgres pg_isready -U aci_user -d aci_memory > /dev/null 2>&1; then
    echo "❌ PostgreSQL is not ready. Waiting longer..."
    sleep 20
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Set environment variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=aci_memory
export DB_USER=aci_user
export DB_PASSWORD=aci_password
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Run database migrations/initialization
echo "🗄️  Initializing database schema..."
python -c "
from src.memory.persistent_memory import get_memory_manager
memory_manager = get_memory_manager()
print('✅ Database connection established')
"

# Seed initial memory if needed
echo "🌱 Seeding initial memory data..."
python scripts/seed_memory.py

echo "🎉 ACI System is ready!"
echo ""
echo "Available commands:"
echo "  python scripts/SeedExperience.py    # Run the main simulation"
echo "  docker-compose logs -f            # View container logs"
echo "  docker-compose down               # Stop containers"
echo ""
echo "Log files are available in the 'logs/' directory with different resolutions:"
echo "  logs/level1/  - Top-level thoughts, actions, world events"
echo "  logs/level2/  - Component interactions, memory operations"
echo "  logs/level3/  - Detailed internal operations, API calls"
echo "  logs/level4/  - Full debug information"
