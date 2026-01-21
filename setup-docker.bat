@echo off
echo 🚀 Setting up Healthcare Chatbot with Docker...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are installed

REM Build and start the services
echo 🔨 Building and starting services...
docker-compose up --build -d

echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check if services are running
echo 🔍 Checking service status...
docker-compose ps

echo 📊 Service URLs:
echo    - Healthcare Chatbot: http://localhost:5000
echo    - Ollama API: http://localhost:11434

echo 🐳 To view logs: docker-compose logs -f
echo 🛑 To stop services: docker-compose down
echo 🔄 To restart services: docker-compose restart

echo ✅ Setup complete! Your healthcare chatbot is now running in Docker containers.
pause 