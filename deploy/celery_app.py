# celery_app.py - Celery application configuration

from celery import Celery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_celery_app():
    """Create and configure Celery application"""
    
    # Redis configuration (default)
    broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # Alternative: RabbitMQ configuration
    # broker_url = os.getenv('CELERY_BROKER_URL', 'pyamqp://guest@localhost//')
    # result_backend = os.getenv('CELERY_RESULT_BACKEND', 'rpc://')
    
    # Create Celery app
    celery_app = Celery(
        'contract_analyzer',
        broker=broker_url,
        backend=result_backend,
        include=[
            'tasks.pdf_processing',
            'tasks.batch_processing',
            'tasks.maintenance'
        ]
    )
    
    # Celery configuration
    celery_app.conf.update(
        # Serialization
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        
        # Timezone
        timezone='UTC',
        enable_utc=True,
        
        # Task execution
        task_track_started=True,
        task_time_limit=30 * 60,        # 30 minutes hard limit
        task_soft_time_limit=25 * 60,   # 25 minutes soft limit
        task_acks_late=True,             # Acknowledge task after completion
        
        # Worker configuration
        worker_prefetch_multiplier=1,    # Prefetch one task at a time
        worker_disable_rate_limits=False,
        worker_pool_restarts=True,
        
        # Compression
        task_compression='gzip',
        result_compression='gzip',
        
        # Result backend settings
        result_expires=3600,            # Results expire after 1 hour
        result_persistent=True,         # Persist results
        
        # Routing (optional - for different queues)
        task_routes={
            'tasks.pdf_processing.process_pdf_async': {'queue': 'pdf_processing'},
            'tasks.batch_processing.process_batch_async': {'queue': 'batch_processing'},
            'tasks.maintenance.cleanup_old_sessions_async': {'queue': 'maintenance'}
        },
        
        # Monitoring
        worker_send_task_events=True,
        task_send_sent_event=True,
    )
    
    return celery_app

# Create the Celery app instance
celery = create_celery_app()

if __name__ == '__main__':
    celery.start()


# docker-compose.yml - For easy Redis and Celery setup

version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  celery_worker:
    build: .
    command: celery -A celery_app worker --loglevel=info --concurrency=4
    volumes:
      - .:/app
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    restart: unless-stopped

  celery_beat:
    build: .
    command: celery -A celery_app beat --loglevel=info
    volumes:
      - .:/app
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    restart: unless-stopped

  flower:
    build: .
    command: celery -A celery_app flower --port=5555
    ports:
      - "5555:5555"
    volumes:
      - .:/app
    depends_on:
      - redis
      - celery_worker
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    restart: unless-stopped

volumes:
  redis_data:


# start_celery.sh - Shell script to start Celery worker

#!/bin/bash

# Start Celery worker
echo "Starting Celery worker..."

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Celery worker with multiple queues
celery -A celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=pdf_processing,batch_processing,maintenance \
    --hostname=worker@%h

# Alternative: Start worker for specific queue only
# celery -A celery_app worker --loglevel=info --queues=pdf_processing


# start_flower.sh - Shell script to start Flower monitoring

#!/bin/bash

# Start Flower monitoring tool
echo "Starting Flower monitoring..."

# Set environment variables if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Flower
celery -A celery_app flower --port=5555 --basic_auth=admin:password

echo "Flower is running at http://localhost:5555"
echo "Login: admin / password"


# tasks/__init__.py - Tasks package initialization

# This file makes the tasks directory a Python package


# tasks/maintenance.py - Maintenance tasks

from celery import Celery
from datetime import datetime, timedelta
from config.database import db

# Import Celery app
from celery_app import celery

@celery.task
def cleanup_old_sessions(hours=24):
    """Clean up old user sessions"""
    
    try:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        sql = "DELETE FROM users WHERE last_active < %s"
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (cutoff_time,))
                deleted_count = cur.rowcount
                conn.commit()
        
        return {
            'success': True,
            'deleted_sessions': deleted_count,
            'message': f'Cleaned up {deleted_count} old sessions'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Session cleanup failed'
        }

@celery.task
def cleanup_old_batch_jobs(days=7):
    """Clean up old completed batch jobs"""
    
    try:
        cutoff_time = datetime.now() - timedelta(days=days)
        
        sql = """
            DELETE FROM batch_jobs 
            WHERE status IN ('completed', 'failed') 
            AND completed_at < %s
        """
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (cutoff_time,))
                deleted_count = cur.rowcount
                conn.commit()
        
        return {
            'success': True,
            'deleted_jobs': deleted_count,
            'message': f'Cleaned up {deleted_count} old batch jobs'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Batch job cleanup failed'
        }

# Periodic task scheduling (optional)
from celery.schedules import crontab

celery.conf.beat_schedule = {
    'cleanup-sessions-daily': {
        'task': 'tasks.maintenance.cleanup_old_sessions',
        'schedule': crontab(hour=2, minute=0),  # Run daily at 2:00 AM
        'args': (24,)  # Clean sessions older than 24 hours
    },
    'cleanup-batch-jobs-weekly': {
        'task': 'tasks.maintenance.cleanup_old_batch_jobs',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),  # Run weekly on Sunday at 3:00 AM
        'args': (7,)  # Clean batch jobs older than 7 days
    },
}


# .env - Updated environment variables template

# Database Configuration
DB_USERNAME=your_aurora_username
DB_PASSWORD=your_secure_password
DB_HOST=your-aurora-cluster.cluster-xxxxxxxxx.us-east-1.rds.amazonaws.com
DB_PORT=5432
DB_NAME=contract_analysis

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Alternative: RabbitMQ configuration
# CELERY_BROKER_URL=pyamqp://guest@localhost//
# CELERY_RESULT_BACKEND=rpc://

# Application Configuration
MAX_FILE_SIZE_MB=10
MAX_BATCH_SIZE=20
SESSION_TIMEOUT_HOURS=24

# Development
SQL_DEBUG=false
