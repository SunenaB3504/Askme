# Deployment Guide - Making Nia's Learning Assistant Accessible Online

## Overview
This guide shows you how to deploy Nia's Learning Assistant so it can be accessed from any computer with an internet connection.

## Option 1: Railway (Recommended - Free & Easy)

### Step 1: Prepare for Deployment
Create a Procfile for Railway:

```bash
# In your project root, create a file named 'Procfile' (no extension)
web: python nia_launcher.py
```

### Step 2: Update requirements.txt
Ensure all dependencies are listed:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
jinja2==3.1.2
python-multipart==0.0.6
```

### Step 3: Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with your GitHub account
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your `SunenaB3504/Askme` repository
5. Railway will automatically detect and deploy your app
6. You'll get a URL like: `https://askme-production.up.railway.app`

## Option 2: Render (Free Tier Available)

### Step 1: Manual Deployment Settings
When creating a new Web Service on Render, use these settings:

**Build Command:**
```bash
pip install -r requirements-web.txt
```

**Start Command:**
```bash
python nia_launcher.py
```

**Alternative Start Command (If Issues):**
```bash
uvicorn nia_launcher:app --host 0.0.0.0 --port $PORT
```

### Step 2: Environment Variables
Set these in Render dashboard:
- `PYTHON_VERSION`: `3.11.0`

### Step 3: Deploy
1. Go to [render.com](https://render.com)
2. Connect your GitHub account
3. Select your repository: `SunenaB3504/Askme`
4. Choose "Web Service"
5. Use the settings above
6. Click "Create Web Service"

### Step 4: Alternative - Using render.yaml (Automatic)
The render.yaml file is configured for lightweight web deployment:

```yaml
services:
  - type: web
    name: nia-learning-assistant
    env: python
    buildCommand: "pip install -r requirements-web.txt"
    startCommand: "python nia_launcher.py"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### Why Use requirements-web.txt?
- ✅ **Lightweight**: Only web dependencies (FastAPI, Uvicorn)
- ✅ **Fast Build**: No heavy AI/ML libraries
- ✅ **Compatible**: Works with Python 3.11+
- ✅ **Cost Effective**: Uses minimal resources

## Option 3: Heroku (Paid but Reliable)

### Step 1: Create Procfile
```
web: python nia_launcher.py
```

### Step 2: Create runtime.txt
```
python-3.9.18
```

### Step 3: Deploy to Heroku
1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create nia-learning-assistant`
4. Deploy: `git push heroku master`

## Option 4: Replit (Good for Testing)

### Step 1: Import to Replit
1. Go to [replit.com](https://replit.com)
2. Click "Create" → "Import from GitHub"
3. Enter your repository URL
4. Replit will automatically set up the environment

### Step 2: Configure
- Main file: `nia_launcher.py`
- Run command: `python nia_launcher.py`

## Option 5: GitHub Codespaces (For Development)

### Step 1: Enable Codespaces
1. Go to your GitHub repository
2. Click "Code" → "Codespaces" → "Create codespace"
3. This creates a cloud development environment

### Step 2: Port Forwarding
- When you run the app, Codespaces will automatically forward the port
- You'll get a URL to access your app

## Quick Setup for Railway (Easiest)

Let me help you prepare the files needed for Railway deployment:

### 1. Create Procfile
```
web: uvicorn nia_launcher:app --host 0.0.0.0 --port $PORT
```

### 2. Update nia_launcher.py for Cloud Deployment
```python
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print("🎓 Starting Nia's Learning Assistant...")
    print("📚 Loaded training data and questions")
    print(f"🌟 Server will be available at: http://0.0.0.0:{port}")
    print("💫 Ready to help Nia learn!")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
```

## Environment Variables (Optional)
For production deployment, you might want to set:

```bash
ENVIRONMENT=production
DEBUG=false
PORT=8000
```

## Security Considerations
- The app currently has no authentication
- Consider adding basic password protection for public deployment
- Educational content is safe to be public

## Cost Comparison
- **Railway**: Free tier (500 hours/month)
- **Render**: Free tier (750 hours/month)
- **Heroku**: $5-7/month (no free tier)
- **Replit**: Free for public projects
- **GitHub Codespaces**: 60 hours free/month

## Recommended Approach
1. **For immediate testing**: Use Replit or GitHub Codespaces
2. **For permanent hosting**: Use Railway or Render
3. **For production use**: Consider Heroku or a VPS

## Next Steps
Choose one of the platforms above and follow the specific deployment steps. The Railway option is recommended for beginners as it requires minimal configuration.

Once deployed, you'll have a URL that anyone can use to access Nia's Learning Assistant from anywhere in the world!
