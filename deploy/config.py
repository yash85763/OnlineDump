"""
Configuration Module for Contract Analysis Application

This module manages configuration settings for the application,
including database credentials, AWS services, and application parameters.
It supports loading configuration from environment variables, config files,
and AWS Secrets Manager.

Usage:
    from config import get_config
    config = get_config()
    db_config = config['database']
"""

import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('config')

# Optional: Import boto3 for AWS Secrets Manager if available
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

def load_config_from_file(config_path):
    """Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the JSON configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {str(e)}")
        return {}

def get_secret(secret_name, region_name="us-east-1"):
    """Retrieve a secret from AWS Secrets Manager.
    
    Args:
        secret_name (str): Name of the secret
        region_name (str): AWS region name
        
    Returns:
        dict: Secret values
    """
    if not BOTO3_AVAILABLE:
        logger.warning("boto3 not available, cannot retrieve secret from AWS Secrets Manager")
        return {}
        
    try:
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        
        response = client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in response:
            return json.loads(response['SecretString'])
        else:
            logger.warning(f"Secret {secret_name} does not contain SecretString")
            return {}
    except ClientError as e:
        logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error retrieving secret {secret_name}: {str(e)}")
        return {}

def get_config():
    """Get the configuration for the application.
    
    Returns:
        dict: Complete configuration dictionary
    """
    # Default configuration
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'contracts_db',
            'user': 'postgres',
            'password': '',
            'min_connections': 1,
            'max_connections': 10
        },
        'aws': {
            'region': 'us-east-1',
            'ec2_instance_id': '',
            's3_bucket': 'contract-analyzer-storage'
        },
        'application': {
            'debug': False,
            'log_level': 'INFO',
            'preloaded_contracts_dir': './preloaded_contracts',
            'temp_dir': './temp',
            'max_file_size_mb': 10,
            'session_ttl_days': 7
        }
    }
    
    # Load from config file if it exists
    config_path = os.environ.get('CONFIG_PATH', 'config.json')
    file_config = load_config_from_file(config_path)
    
    # Deep merge file_config into config
    for section, section_config in file_config.items():
        if section in config and isinstance(config[section], dict) and isinstance(section_config, dict):
            config[section].update(section_config)
        else:
            config[section] = section_config
    
    # Override with environment variables
    # Database settings
    if os.environ.get('DB_HOST'):
        config['database']['host'] = os.environ.get('DB_HOST')
    if os.environ.get('DB_PORT'):
        config['database']['port'] = int(os.environ.get('DB_PORT'))
    if os.environ.get('DB_NAME'):
        config['database']['dbname'] = os.environ.get('DB_NAME')
    if os.environ.get('DB_USER'):
        config['database']['user'] = os.environ.get('DB_USER')
    if os.environ.get('DB_PASSWORD'):
        config['database']['password'] = os.environ.get('DB_PASSWORD')
    
    # AWS settings
    if os.environ.get('AWS_REGION'):
        config['aws']['region'] = os.environ.get('AWS_REGION')
    if os.environ.get('S3_BUCKET'):
        config['aws']['s3_bucket'] = os.environ.get('S3_BUCKET')
    
    # Application settings
    if os.environ.get('DEBUG'):
        config['application']['debug'] = os.environ.get('DEBUG').lower() in ('true', 'yes', '1')
    if os.environ.get('LOG_LEVEL'):
        config['application']['log_level'] = os.environ.get('LOG_LEVEL')
    
    # Try to load database credentials from AWS Secrets Manager if specified
    db_secret_name = os.environ.get('DB_SECRET_NAME')
    if db_secret_name and BOTO3_AVAILABLE:
        logger.info(f"Attempting to load database credentials from secret: {db_secret_name}")
        secret = get_secret(db_secret_name, region_name=config['aws']['region'])
        if secret:
            # Update database config with values from the secret
            if 'host' in secret:
                config['database']['host'] = secret['host']
            if 'port' in secret:
                config['database']['port'] = int(secret['port'])
            if 'dbname' in secret:
                config['database']['dbname'] = secret['dbname']
            if 'username' in secret:
                config['database']['user'] = secret['username']
            if 'password' in secret:
                config['database']['password'] = secret['password']
    
    # Validate essential configuration
    if not config['database']['password']:
        logger.warning("Database password not set!")
    
    return config

def init_directories(config):
    """Initialize application directories based on configuration.
    
    Args:
        config (dict): Application configuration
    """
    # Create required directories if they don't exist
    directories = [
        config['application']['preloaded_contracts_dir'],
        config['application']['temp_dir'],
        f"{config['application']['preloaded_contracts_dir']}/pdfs",
        f"{config['application']['preloaded_contracts_dir']}/jsons"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

# If running this file directly, print the configuration
if __name__ == "__main__":
    config = get_config()
    print(json.dumps(config, indent=4))
    
    # Initialize directories
    init_directories(config)
