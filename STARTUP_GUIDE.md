# BasicChat Startup Guide

This guide explains how to set up automatic startup and graceful shutdown for BasicChat with Redis integration.

## 🚀 Quick Start

### Option 1: Enhanced Startup Script (Recommended)
```bash
# Make scripts executable
chmod +x start_basicchat.sh launch_basicchat.sh

# Start BasicChat with automatic Redis management
./start_basicchat.sh
```

### Option 2: Simple Launcher
```bash
# Use the simple launcher
./launch_basicchat.sh
```

## 📋 Features

### ✅ **Automatic Redis Management**
- Detects if Redis is already running
- Starts Redis automatically if not running
- Supports multiple Redis installation methods:
  - Direct `redis-server` command
  - Homebrew services (macOS)
  - systemctl (Linux)
- Graceful Redis shutdown on exit

### ✅ **Service Health Monitoring**
- Port availability checking
- Service readiness verification
- Automatic retry logic
- Clear status reporting with colors

### ✅ **Graceful Shutdown**
- Handles Ctrl+C gracefully
- Stops all Celery workers
- Terminates Streamlit process
- Cleans up Redis if started by script
- Waits for processes to finish

### ✅ **Environment Management**
- Automatic virtual environment activation
- Environment variable configuration
- Directory setup
- Service dependency checking

## 🔧 Installation Options

### macOS (Homebrew)
```bash
# Install Redis
brew install redis

# Start Redis service (optional - script will handle this)
brew services start redis
```

### Linux (Ubuntu/Debian)
```bash
# Install Redis
sudo apt-get update
sudo apt-get install redis-server

# Start Redis service
sudo systemctl start redis
sudo systemctl enable redis
```

### Manual Installation
```bash
# Download and compile Redis
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make
sudo make install
```

## 🎯 Usage Scenarios

### Development Mode
```bash
# Start with full background task support
./start_basicchat.sh
```

### Production Mode
```bash
# Use Docker Compose for production
docker-compose up --build
```

### Simple Mode (No Redis)
```bash
# Start without background tasks
streamlit run app.py
```

## 🔄 Automatic Startup

### macOS (LaunchAgent)
1. Edit `com.basicchat.startup.plist`:
   ```bash
   # Replace YOUR_USERNAME with your actual username
   sed -i '' 's/YOUR_USERNAME/'$USER'/g' com.basicchat.startup.plist
   ```

2. Install the LaunchAgent:
   ```bash
   cp com.basicchat.startup.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.basicchat.startup.plist
   ```

3. Start the service:
   ```bash
   launchctl start com.basicchat.startup
   ```

### Linux (systemd)
1. Edit `basicchat.service`:
   ```bash
   # Replace paths and username
   sed -i 's|YOUR_USERNAME|'$USER'|g' basicchat.service
   sed -i 's|/path/to/basic-chat|'$(pwd)'|g' basicchat.service
   ```

2. Install the service:
   ```bash
   sudo cp basicchat.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable basicchat.service
   ```

3. Start the service:
   ```bash
   sudo systemctl start basicchat.service
   ```

## 🛠️ Configuration

### Environment Variables
Create `basicchat.env` for custom configuration:
```bash
# Copy the template
cp basicchat.env.example basicchat.env

# Edit configuration
nano basicchat.env
```

### Port Configuration
Default ports (can be changed in `basicchat.env`):
- **Streamlit App**: 8501
- **Redis**: 6379
- **Flower Monitor**: 5555
- **Ollama**: 11434

## 🔍 Monitoring

### Service Status
```bash
# Check if services are running
./start_basicchat.sh --status

# View logs
tail -f basicchat.log
tail -f basicchat_error.log
```

### Web Interfaces
- **Main App**: http://localhost:8501
- **Task Monitor**: http://localhost:5555
- **Redis CLI**: `redis-cli`

## 🚨 Troubleshooting

### Common Issues

#### Redis Connection Refused
```bash
# Check if Redis is running
redis-cli ping

# Start Redis manually
redis-server --port 6379 --daemonize yes
```

#### Port Already in Use
```bash
# Check what's using the port
lsof -i :8501

# Kill the process
kill -9 <PID>
```

#### Permission Issues
```bash
# Make scripts executable
chmod +x *.sh

# Check file permissions
ls -la *.sh
```

### Log Files
- **Application Log**: `app.log`
- **Startup Log**: `basicchat.log`
- **Error Log**: `basicchat_error.log`
- **Redis Log**: Check system logs or `redis_data/redis.log`

## 🔧 Advanced Configuration

### Custom Redis Configuration
```bash
# Create custom Redis config
cat > redis.conf << EOF
port 6379
dir ./redis_data
appendonly yes
maxmemory 256mb
maxmemory-policy allkeys-lru
EOF

# Start with custom config
redis-server redis.conf
```

### Multiple Instances
```bash
# Start multiple BasicChat instances on different ports
REDIS_PORT=6380 STREAMLIT_PORT=8502 ./start_basicchat.sh
```

## 📊 Performance Tips

### Redis Optimization
- Enable persistence with `appendonly yes`
- Set appropriate `maxmemory` limits
- Use `maxmemory-policy allkeys-lru` for caching
- Monitor memory usage with `redis-cli info memory`

### Celery Optimization
- Adjust worker concurrency based on CPU cores
- Use separate queues for different task types
- Monitor task queue length in Flower
- Set appropriate task timeouts

## 🎉 Success Indicators

When everything is working correctly, you should see:
- ✅ All services started successfully
- ✅ Redis responding on port 6379
- ✅ Streamlit app accessible at http://localhost:8501
- ✅ Flower monitoring at http://localhost:5555
- ✅ Graceful shutdown on Ctrl+C

## 📞 Support

If you encounter issues:
1. Check the log files for error messages
2. Verify all dependencies are installed
3. Ensure ports are not in use by other applications
4. Check system resources (CPU, memory, disk space) 
