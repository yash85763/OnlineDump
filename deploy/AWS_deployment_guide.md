# AWS Deployment Guide for Contract Analysis Application

This guide provides instructions for deploying the Contract Analysis application on AWS EC2 with Aurora PostgreSQL database integration.

## Prerequisites

- AWS account with appropriate permissions
- Basic knowledge of AWS services, Linux, and PostgreSQL
- Domain name (optional, for HTTPS setup)

## Step 1: Set Up Aurora PostgreSQL Database

### Create an Aurora PostgreSQL cluster

1. Log in to the AWS Management Console
2. Navigate to Amazon RDS
3. Click "Create database"
4. Choose "Standard create"
5. Select "Amazon Aurora" as the engine type
6. Select "Amazon Aurora PostgreSQL-Compatible Edition"
7. Choose the latest PostgreSQL version (at least 14.x)
8. Select "Production" for Template
9. Configure cluster settings:
   - Cluster identifier: `contract-analyzer-db`
   - Master username: `dbadmin` (remember this)
   - Master password: Create a secure password (remember this)
10. Under instance configuration:
    - Select "Burstable classes" and "db.t4g.medium" for development/testing
    - Select "Memory optimized" and "db.r6g.large" for production
11. Under Availability & durability:
    - Development/testing: Choose "Don't create an Aurora Replica"
    - Production: Choose "Create an Aurora Replica" for high availability
12. Under Connectivity:
    - Create a new VPC or use an existing one
    - Create a new security group
    - Set Public access to "No" for security
13. Under Additional configuration:
    - Initial database name: `contracts_db`
    - Enable automated backups
    - Set backup retention period (7-35 days recommended)
14. Click "Create database"

### Configure Security Group

1. Navigate to EC2 > Security Groups
2. Find the security group created for your Aurora cluster
3. Add a rule that allows PostgreSQL traffic (port 5432) from your EC2 instance security group (you'll configure this later)

## Step 2: Set Up EC2 Instance

### Launch an EC2 Instance

1. Navigate to EC2 in the AWS Management Console
2. Click "Launch instance"
3. Configure the instance:
   - Name: `contract-analyzer-app`
   - AMI: Amazon Linux 2023 or Ubuntu 22.04 LTS
   - Instance type: t2.medium recommended (2 vCPU, 4 GiB Memory)
   - Create or select a key pair for SSH access
   - Create a new security group or use an existing one
   - Allow HTTP (80), HTTPS (443), and SSH (22) inbound traffic
4. Launch the instance

### Configure Security Group

1. Go to EC2 > Security Groups
2. Find the security group for your EC2 instance
3. Add inbound rules:
   - SSH (port 22) from your IP address
   - HTTP (port 80) from anywhere (0.0.0.0/0)
   - HTTPS (port 443) from anywhere (0.0.0.0/0)
   - Streamlit (port 8501) from anywhere or restrict to your IP

## Step 3: Configure the EC2 Instance

### Connect to the Instance

```bash
ssh -i /path/to/your-key.pem ec2-user@your-instance-public-ip
```

### Install Required Software

For Amazon Linux 2023:

```bash
# Update system
sudo dnf update -y

# Install git, Python, and PostgreSQL client
sudo dnf install -y git python3 python3-pip python3-devel postgresql15 postgresql15-devel gcc

# Install system dependencies for PDF processing
sudo dnf install -y poppler-utils

# Create application directory
sudo mkdir -p /opt/contract-analyzer
sudo chown ec2-user:ec2-user /opt/contract-analyzer

# Create directories for uploaded and processed files
mkdir -p /opt/contract-analyzer/preloaded_contracts/pdfs
mkdir -p /opt/contract-analyzer/preloaded_contracts/jsons
mkdir -p /opt/contract-analyzer/temp
```

For Ubuntu 22.04:

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install git, Python, and PostgreSQL client
sudo apt install -y git python3 python3-pip python3-dev libpq-dev build-essential

# Install system dependencies for PDF processing
sudo apt install -y poppler-utils

# Create application directory
sudo mkdir -p /opt/contract-analyzer
sudo chown ubuntu:ubuntu /opt/contract-analyzer

# Create directories for uploaded and processed files
mkdir -p /opt/contract-analyzer/preloaded_contracts/pdfs
mkdir -p /opt/contract-analyzer/preloaded_contracts/jsons
mkdir -p /opt/contract-analyzer/temp
```

## Step 4: Clone and Set Up Application

### Clone the Repository

```bash
cd /opt/contract-analyzer
git clone https://github.com/yourusername/contract-analyzer.git .
```

Alternatively, you can upload your code using SFTP or SCP:

```bash
# From your local machine
scp -i /path/to/your-key.pem -r /path/to/your/project/* ec2-user@your-instance-public-ip:/opt/contract-analyzer/
```

### Create Virtual Environment and Install Dependencies

```bash
cd /opt/contract-analyzer
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install streamlit psycopg2-binary pdfminer.six PyPDF2 scikit-learn boto3 sentence-transformers faiss-cpu
pip install -r requirements.txt  # If you have a requirements.txt file
```

### Configure Application Settings

Create a configuration file with your database credentials:

```bash
cd /opt/contract-analyzer
cat > config.json << EOL
{
  "database": {
    "host": "your-aurora-cluster-endpoint.rds.amazonaws.com",
    "port": 5432,
    "dbname": "contracts_db",
    "user": "dbadmin",
    "password": "your-secure-password",
    "min_connections": 1,
    "max_connections": 10
  },
  "aws": {
    "region": "us-east-1",
    "s3_bucket": "contract-analyzer-storage"
  },
  "application": {
    "debug": false,
    "log_level": "INFO",
    "preloaded_contracts_dir": "/opt/contract-analyzer/preloaded_contracts",
    "temp_dir": "/opt/contract-analyzer/temp"
  }
}
EOL

# Secure the config file
chmod 600 config.json
```

Alternatively, you can use environment variables (more secure):

```bash
# Create an environment file
cat > .env << EOL
DB_HOST=your-aurora-cluster-endpoint.rds.amazonaws.com
DB_PORT=5432
DB_NAME=contracts_db
DB_USER=dbadmin
DB_PASSWORD=your-secure-password
AWS_REGION=us-east-1
S3_BUCKET=contract-analyzer-storage
EOL

# Secure the env file
chmod 600 .env
```

## Step 5: Test the Application

Test the application locally on the EC2 instance:

```bash
cd /opt/contract-analyzer
source venv/bin/activate

# Load environment variables if using .env file
export $(grep -v '^#' .env | xargs)

# Run the Streamlit app
streamlit run streamlit_app.py
```

You should see output indicating that Streamlit is running, typically on port 8501.

## Step 6: Set Up Service for Production

Create a systemd service for running the application:

```bash
sudo tee /etc/systemd/system/contract-analyzer.service << EOL
[Unit]
Description=Contract Analyzer Streamlit Application
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/opt/contract-analyzer
ExecStart=/opt/contract-analyzer/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=on-failure
RestartSec=5s
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=contract-analyzer
Environment="LC_ALL=en_US.UTF-8"
Environment="LANG=en_US.UTF-8"
# Include environment variables from file
EnvironmentFile=/opt/contract-analyzer/.env

[Install]
WantedBy=multi-user.target
EOL
```

For Ubuntu, change `User=ec2-user` to `User=ubuntu`.

Start and enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl start contract-analyzer
sudo systemctl enable contract-analyzer

# Check status
sudo systemctl status contract-analyzer
```

## Step 7: Set Up Nginx as a Reverse Proxy (Optional but Recommended)

Install and configure Nginx to serve your application:

```bash
# For Amazon Linux
sudo dnf install -y nginx

# For Ubuntu
sudo apt install -y nginx
```

Create an Nginx configuration file:

```bash
sudo tee /etc/nginx/conf.d/contract-analyzer.conf << EOL
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain or public IP

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header Host \$host;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
EOL
```

Start and enable Nginx:

```bash
sudo systemctl start nginx
sudo systemctl enable nginx
```

## Step 8: Set Up HTTPS with Let's Encrypt (Optional)

If you have a domain name pointing to your EC2 instance, you can set up HTTPS:

```bash
# For Amazon Linux
sudo dnf install -y certbot python3-certbot-nginx

# For Ubuntu
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Follow the prompts and certbot will automatically configure Nginx
```

## Step 9: Set Up Database Initialization Script

Create a script to initialize the database schema:

```bash
cd /opt/contract-analyzer
cat > init_db.py << EOL
#!/usr/bin/env python3

from db_handler import DatabaseHandler
from config import get_config

def main():
    print("Initializing database schema...")
    config = get_config()
    db = DatabaseHandler(config['database'])
    db.initialize_schema()
    db.close()
    print("Database schema initialized successfully.")

if __name__ == "__main__":
    main()
EOL

chmod +x init_db.py
```

Run the initialization script:

```bash
cd /opt/contract-analyzer
source venv/bin/activate
python init_db.py
```

## Step 10: Set Up Backup and Monitoring

### Set Up Database Backups

Aurora PostgreSQL automatically creates backups based on the retention period you set. For additional backup options:

1. Navigate to RDS > Databases > your-database
2. Click "Actions" > "Take snapshot" to create manual snapshots
3. Consider setting up automated snapshot exports to S3

### Set Up CloudWatch Monitoring

1. Navigate to CloudWatch in the AWS Management Console
2. Set up alarms for:
   - EC2 instance CPU utilization
   - EC2 instance memory utilization
   - RDS database connections
   - RDS CPU utilization
   - RDS free storage space

### Set Up Application Logging

Modify your application to log to CloudWatch:

1. Install the AWS CloudWatch Logs agent on your EC2 instance
2. Configure it to send your application logs to CloudWatch

```bash
# Install CloudWatch agent
sudo dnf install -y amazon-cloudwatch-agent  # Amazon Linux
sudo apt install -y amazon-cloudwatch-agent  # Ubuntu

# Configure CloudWatch agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard

# Start the CloudWatch agent
sudo systemctl start amazon-cloudwatch-agent
sudo systemctl enable amazon-cloudwatch-agent
```

## Step 11: Load Test Data (Optional)

To load pre-loaded contracts:

1. Upload your PDF files to the EC2 instance:

```bash
# From your local machine
scp -i /path/to/your-key.pem /path/to/your/pdfs/* ec2-user@your-instance-public-ip:/opt/contract-analyzer/preloaded_contracts/pdfs/

# From your local machine
scp -i /path/to/your-key.pem /path/to/your/jsons/* ec2-user@your-instance-public-ip:/opt/contract-analyzer/preloaded_contracts/jsons/
```

2. Set proper permissions:

```bash
# On your EC2 instance
cd /opt/contract-analyzer
chmod -R 755 preloaded_contracts
```

## Common Issues and Troubleshooting

### Database Connection Issues

If you're having trouble connecting to the Aurora PostgreSQL database:

1. Verify security group settings to ensure your EC2 instance can access the database
2. Check that the database endpoint, username, and password are correct
3. Test the connection using the `psql` command:

```bash
psql -h your-aurora-cluster-endpoint.rds.amazonaws.com -U dbadmin -d contracts_db
```

### Streamlit Application Not Starting

If the Streamlit application fails to start:

1. Check the logs: `sudo journalctl -u contract-analyzer`
2. Verify that all dependencies are installed: `pip install -r requirements.txt`
3. Ensure the configuration file has correct settings

### Web Interface Not Loading

If you can't access the web interface:

1. Check that the Streamlit application is running: `sudo systemctl status contract-analyzer`
2. Verify that the EC2 security group allows traffic on port 8501 (or 80/443 if using Nginx)
3. If using Nginx, check its status: `sudo systemctl status nginx`
4. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`

## Maintenance Tasks

### Updating the Application

To update the application code:

```bash
cd /opt/contract-analyzer
git pull  # If you've cloned a repository

# Or upload new files using SCP/SFTP

# Restart the application
sudo systemctl restart contract-analyzer
```

### Database Maintenance

To perform database maintenance:

1. Navigate to RDS > Databases > your-database
2. For minor version upgrades, AWS can automatically apply these during maintenance windows
3. For major version upgrades, you'll need to manually initiate the process

### Scaling

As your application grows:

1. For vertical scaling, you can resize your EC2 instance to a larger type
2. For horizontal scaling, consider using AWS Elastic Beanstalk or ECS with a load balancer
3. For the database, Aurora can automatically scale storage and you can adjust instance size as needed

## Security Best Practices

1. **Use AWS Secrets Manager** for storing database credentials instead of config files
2. **Implement IAM roles** for EC2 to securely access AWS services without storing credentials
3. **Enable VPC flow logs** to monitor network traffic
4. **Use Security Groups** to restrict access to only necessary ports
5. **Keep your software updated** with regular system updates
6. **Set up AWS CloudTrail** to log all API activity for auditing
7. **Backup your data** regularly and test restores
8. **Implement AWS WAF** if your application is publicly accessible
9. **Use private subnets** for your database and application services when possible

## Conclusion

You've now deployed your Contract Analysis application on AWS EC2 with Aurora PostgreSQL. This setup provides a scalable, reliable, and secure environment for your application. Monitor your resources regularly, keep your software updated, and implement proper backup strategies to ensure continued operation.
