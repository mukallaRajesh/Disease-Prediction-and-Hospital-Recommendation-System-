# Healthcare Chatbot - Hugging Face Spaces Deployment

This is the Hugging Face Spaces version of the Healthcare Chatbot application.

## 🚀 Quick Start

1. **Fork this repository** to your Hugging Face account
2. **Create a new Space** on Hugging Face Spaces
3. **Choose "Gradio"** as the SDK
4. **Upload all files** to your Space
5. **Use `app_hf.py`** as the main application file

## 📁 Required Files for HF Spaces

### Core Files:
- `app_hf.py` - Main Flask application (HF optimized)
- `requirements_hf.txt` - Dependencies (no Ollama)
- All CSV data files
- All ML model files (.pkl)
- `templates/` folder
- `static/` folder

### Files to Exclude:
- `app.py` (use `app_hf.py` instead)
- `requirements.txt` (use `requirements_hf.txt`)
- `docker-compose.yml`
- `Dockerfile`
- `start-ollama.sh`

## 🔧 Key Changes for HF Spaces

### 1. Cache Directory Fix
```python
# Uses temp directory instead of root
cache_dir = os.path.join(tempfile.gettempdir(), 'hf_cache')
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
```

### 2. Database Path Fix
```python
# Uses temp directory for database
db_path = os.path.join(tempfile.gettempdir(), 'users.db')
```

### 3. Port Configuration
```python
# HF Spaces uses port 7860
app.run(debug=False, host='0.0.0.0', port=7860)
```

### 4. No Ollama Dependency
- Removed Ollama integration
- Uses enhanced responses from knowledge base
- Still provides intelligent chatbot responses

## 🎯 Features Available on HF Spaces

✅ **Disease Prediction** - ML models work perfectly  
✅ **Health Quizzes** - Heart and lung assessments  
✅ **User Authentication** - Signup/login system  
✅ **Chatbot** - Intelligent responses from knowledge base  
✅ **User Profiles** - History and statistics  
✅ **Beautiful UI** - 3D avatar and modern design  

## 📊 Deployment Steps

1. **Create Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" SDK
   - Set visibility (Public/Private)

2. **Upload Files:**
   - Upload `app_hf.py` as main file
   - Upload `requirements_hf.txt`
   - Upload all CSV files
   - Upload all .pkl model files
   - Upload `templates/` folder
   - Upload `static/` folder

3. **Configure Space:**
   - Set Python version to 3.11
   - Set main file to `app_hf.py`
   - Set requirements to `requirements_hf.txt`

## 🚨 Important Notes

### Version Compatibility:
- **scikit-learn 1.6.1** - Matches your trained models
- **sentence-transformers 2.2.2** - Stable version
- **torch 2.0.1** - Compatible with transformers

### Cache Permissions:
- Uses temporary directory for cache
- No root directory access needed
- Automatic cleanup on restart

### Database:
- Uses temporary SQLite database
- Data persists during session
- Resets on Space restart

## 🌐 Access Your Application

Once deployed, your application will be available at:
`https://your-username-your-space-name.hf.space`

## 🔍 Troubleshooting

### Common Issues:

1. **Cache Permission Error:**
   - ✅ Fixed with temp directory
   - ✅ Uses `cache_dir` configuration

2. **Model Version Mismatch:**
   - ✅ Fixed with scikit-learn 1.6.1
   - ✅ Compatible with your trained models

3. **Port Issues:**
   - ✅ Uses port 7860 for HF Spaces
   - ✅ Configured in `app_hf.py`

4. **Database Issues:**
   - ✅ Uses temp directory
   - ✅ No permission problems

## 🎉 Success!

Your healthcare chatbot will work perfectly on Hugging Face Spaces with:
- ✅ All ML functionality
- ✅ User authentication
- ✅ Health assessments
- ✅ Intelligent chatbot
- ✅ Beautiful UI
- ✅ No permission issues

The only difference from Docker version is no Ollama AI enhancement, but the core functionality is identical! 