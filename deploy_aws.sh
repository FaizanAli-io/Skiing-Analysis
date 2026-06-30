#!/bin/bash
# AWS Deployment Script for Skiing Analysis Platform
# Run this script on your EC2 instance after getting credentials from client

set -e  # Exit on any error

echo "======================================"
echo "Skiing Analysis - AWS Deployment"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu
if [ ! -f /etc/lsb-release ]; then
    echo -e "${RED}Error: This script is designed for Ubuntu systems${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

echo -e "${GREEN}Step 2: Installing system dependencies...${NC}"
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    postgresql-client \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    awscli \
    nginx \
    git

echo -e "${GREEN}Step 3: Setting up application directory...${NC}"
APP_DIR="/home/ubuntu/skiing-analysis"

if [ ! -d "$APP_DIR" ]; then
    echo "Creating application directory..."
    mkdir -p "$APP_DIR"
fi

cd "$APP_DIR"

echo -e "${GREEN}Step 4: Creating Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi

source venv/bin/activate

echo -e "${GREEN}Step 5: Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${YELLOW}Step 6: Environment configuration${NC}"
echo "Please ensure you have created a .env file with the following variables:"
echo ""
echo "DATABASE_URL=postgresql://username:password@rds-endpoint:5432/dbname"
echo "AWS_ACCESS_KEY_ID=your_access_key"
echo "AWS_SECRET_ACCESS_KEY=your_secret_key"
echo "AWS_REGION=us-east-1"
echo "AWS_S3_BUCKET=skiing-analysis-production"
echo "OPENAI_API=your_openai_key"
echo "JWT_SECRET=your_jwt_secret"
echo "DEFAULT_ADMIN_EMAIL=admin@example.com"
echo "DEFAULT_ADMIN_PASSWORD=secure_password"
echo "DEFAULT_ADMIN_NAME=Bluerun Admin"
echo ""

read -p "Have you created the .env file? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Please create .env file and run this script again${NC}"
    exit 1
fi

echo -e "${GREEN}Step 7: Testing database connection...${NC}"
if [ -f ".env" ]; then
    source .env
    echo "Testing connection to: $DATABASE_URL"
    # Extract host and credentials for testing
    # This is a simple test, actual connection will be validated by the app
    echo -e "${YELLOW}Database connection will be validated when app starts${NC}"
else
    echo -e "${RED}Error: .env file not found!${NC}"
    exit 1
fi

echo -e "${GREEN}Step 8: Running database migrations...${NC}"
python3 << EOF
from database import Base, engine, ensure_database_schema
from main import ensure_default_admin

print("Creating database tables...")
Base.metadata.create_all(engine)

print("Running schema updates...")
ensure_database_schema()

print("Creating default admin user...")
ensure_default_admin()

print("Database setup complete!")
EOF

echo -e "${GREEN}Step 9: Testing AWS S3 connection...${NC}"
aws configure list
echo "Testing S3 access..."
aws s3 ls s3://$AWS_S3_BUCKET/ > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}S3 connection successful!${NC}"
else
    echo -e "${YELLOW}Warning: Could not access S3 bucket. Check credentials.${NC}"
fi

echo -e "${GREEN}Step 10: Creating necessary directories...${NC}"
mkdir -p outputs
mkdir -p temp_videos
mkdir -p logs

echo -e "${GREEN}Step 11: Setting up systemd service...${NC}"
sudo tee /etc/systemd/system/skiing-analysis.service > /dev/null << EOF
[Unit]
Description=Skiing Analysis FastAPI Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}Step 12: Starting the application service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable skiing-analysis
sudo systemctl start skiing-analysis

echo -e "${GREEN}Step 13: Checking application status...${NC}"
sleep 3
sudo systemctl status skiing-analysis --no-pager

echo -e "${GREEN}Step 14: Setting up nginx reverse proxy...${NC}"
sudo tee /etc/nginx/sites-available/skiing-analysis > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long video processing
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/skiing-analysis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

echo ""
echo -e "${GREEN}======================================"
echo "Deployment Complete!"
echo "======================================${NC}"
echo ""
echo "Your application is now running at:"
echo -e "${GREEN}http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)${NC}"
echo ""
echo "Useful commands:"
echo "  - View logs: sudo journalctl -u skiing-analysis -f"
echo "  - Restart app: sudo systemctl restart skiing-analysis"
echo "  - Check status: sudo systemctl status skiing-analysis"
echo "  - Stop app: sudo systemctl stop skiing-analysis"
echo ""
echo "Next steps:"
echo "  1. Test API: curl http://localhost:8000/docs"
echo "  2. Deploy frontend to S3 or serve via nginx"
echo "  3. Setup SSL certificate with Let's Encrypt"
echo "  4. Configure CloudWatch logging"
echo ""
echo -e "${YELLOW}Security reminder:${NC}"
echo "  - Update security group to restrict port 22 to your IP only"
echo "  - Setup CloudWatch alarms for monitoring"
echo "  - Enable S3 versioning and lifecycle policies"
echo "  - Rotate IAM credentials regularly"
echo ""
