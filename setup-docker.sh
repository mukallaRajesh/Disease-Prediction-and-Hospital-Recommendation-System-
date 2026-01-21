#!/bin/bash

echo "🚀 Setting up Healthcare Chatbot with Docker..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Build and start the services
echo "🔨 Building and starting services..."
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

echo "📊 Service URLs:"
echo "   - Healthcare Chatbot: http://localhost:5000"
echo "   - Ollama API: http://localhost:11434"

echo "🐳 To view logs: docker-compose logs -f"
echo "🛑 To stop services: docker-compose down"
echo "🔄 To restart services: docker-compose restart"

echo "✅ Setup complete! Your healthcare chatbot is now running in Docker containers." 