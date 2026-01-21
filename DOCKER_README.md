# Healthcare Chatbot - Docker Deployment

This guide will help you deploy your healthcare chatbot application using Docker containers, including the Ollama server for AI responses.

## 🐳 Prerequisites

1. **Docker Desktop** - Download and install from [Docker's official website](https://www.docker.com/products/docker-desktop/)
2. **Docker Compose** - Usually comes with Docker Desktop
3. **At least 8GB RAM** - Required for running Ollama models
4. **At least 10GB free disk space** - For Docker images and Ollama models

## 🚀 Quick Start

### Option 1: Using Setup Scripts

**For Linux/Mac:**
```bash
chmod +x setup-docker.sh
./setup-docker.sh
```

**For Windows:**
```cmd
setup-docker.bat
```

### Option 2: Manual Setup

1. **Build and start the services:**
   ```bash
   docker-compose up --build -d
   ```

2. **Wait for services to start (about 30-60 seconds):**
   ```bash
   docker-compose ps
   ```

3. **Access your application:**
   - Healthcare Chatbot: http://localhost:5000
   - Ollama API: http://localhost:11434

## 📋 Service Details

### Healthcare Chatbot Container
- **Port:** 5000
- **Image:** Built from your application code
- **Features:** Flask web application with disease prediction and chatbot
- **Database:** SQLite (persisted via volume)

### Ollama Container
- **Port:** 11434
- **Image:** ollama/ollama:latest
- **Features:** AI model server for chatbot responses
- **Models:** Automatically downloads required models

## 🔧 Configuration

### Environment Variables

You can modify the `docker-compose.yml` file to change:

- **Port mappings:** Change the port numbers if needed
- **Ollama host:** The URL where Ollama is accessible
- **Volume paths:** Where data is persisted

### Model Configuration

The application uses the `llama3-chatqa` model. To use a different model:

1. Edit `app.py` line ~350:
   ```python
   response = ollama.chat(
       model="your-model-name",  # Change this
       messages=[{"role": "user", "content": prompt}]
   )
   ```

2. Pull the new model:
   ```bash
   docker-compose exec ollama ollama pull your-model-name
   ```

## 📊 Monitoring and Logs

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f healthcare-chatbot
docker-compose logs -f ollama
```

### Check Service Status
```bash
docker-compose ps
```

### Resource Usage
```bash
docker stats
```

## 🛠️ Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :5000
   # Or change ports in docker-compose.yml
   ```

2. **Ollama model not found:**
   ```bash
   # Pull the model manually
   docker-compose exec ollama ollama pull llama3-chatqa
   ```

3. **Out of memory:**
   - Increase Docker memory limit in Docker Desktop settings
   - Use a smaller model or reduce batch size

4. **Database issues:**
   ```bash
   # Reset database (WARNING: This will delete all data)
   docker-compose down
   rm users.db
   docker-compose up -d
   ```

### Health Checks

The services include health checks. Check their status:
```bash
docker-compose ps
```

## 🔄 Management Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

### Rebuild Services (after code changes)
```bash
docker-compose up --build -d
```

### Update Ollama
```bash
docker-compose pull ollama
docker-compose up -d ollama
```

## 📁 File Structure

```
healthcare-chatbot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container setup
├── .dockerignore         # Files to exclude from build
├── setup-docker.sh       # Linux/Mac setup script
├── setup-docker.bat      # Windows setup script
├── templates/            # HTML templates
├── static/              # CSS, JS, images
└── *.csv                # Data files
```

## 🔒 Security Considerations

1. **Change the secret key** in `app.py`:
   ```python
   app.secret_key = 'your-secure-secret-key-here'
   ```

2. **Use environment variables** for sensitive data:
   ```yaml
   environment:
     - SECRET_KEY=your-secret-key
   ```

3. **Limit network access** in production:
   ```yaml
   networks:
     - internal-network
   ```

## 🚀 Production Deployment

For production deployment:

1. **Use a reverse proxy** (nginx/traefik)
2. **Enable HTTPS** with SSL certificates
3. **Set up monitoring** (Prometheus/Grafana)
4. **Configure backups** for the database
5. **Use Docker secrets** for sensitive data

## 📞 Support

If you encounter issues:

1. Check the logs: `docker-compose logs -f`
2. Verify Docker is running: `docker --version`
3. Check available disk space: `df -h`
4. Monitor resource usage: `docker stats`

## 🎉 Success!

Once everything is running, you should see:
- ✅ Healthcare Chatbot accessible at http://localhost:5000
- ✅ Ollama API running at http://localhost:11434
- ✅ All services showing as "Up" in `docker-compose ps`

Your healthcare chatbot is now fully containerized and ready to use! 🎊 