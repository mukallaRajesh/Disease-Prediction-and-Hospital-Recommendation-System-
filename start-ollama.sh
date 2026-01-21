#!/bin/bash

# Start Ollama server in background
echo "🚀 Starting Ollama server..."
ollama serve &

# Wait for server to be ready
echo "⏳ Waiting for Ollama server to start..."
sleep 15

# Check if model exists, if not pull it
echo "🔍 Checking if llama3-chatqa model exists..."
if ! ollama list | grep -q "llama3-chatqa"; then
    echo "📥 Pulling llama3-chatqa model (this may take a few minutes)..."
    ollama pull llama3-chatqa
    echo "✅ Model downloaded successfully!"
else
    echo "✅ Model already exists!"
fi

# List available models
echo "📋 Available models:"
ollama list

# Keep the container running
echo "🎯 Ollama is ready! Keeping container alive..."
wait 