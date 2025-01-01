"""
Demonstration of Flask async support and background tasks.
"""

from flask import Flask, jsonify, request
from celery import Celery
import asyncio
from asgiref.wsgi import WsgiToAsgi
from quart import websocket
import aiohttp
import redis
from rq import Queue
from rq.job import Job
import threading
import time
from functools import wraps
import logging

app = Flask(__name__)

# Celery configuration
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',
    CELERY_TASK_SERIALIZER='json',
    CELERY_ACCEPT_CONTENT=['json'],
    CELERY_RESULT_SERIALIZER='json',
    CELERY_TIMEZONE='UTC',
    CELERY_ENABLE_UTC=True
)

# Initialize Celery
celery = Celery(
    app.name,
    broker=app.config['CELERY_BROKER_URL'],
    backend=app.config['CELERY_RESULT_BACKEND']
)
celery.conf.update(app.config)

# Initialize Redis Queue
redis_conn = redis.Redis()
task_queue = Queue(connection=redis_conn)

# Async helper
def async_task(f):
    """Decorator to run a function asynchronously."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        def task():
            with app.app_context():
                f(*args, **kwargs)
        thread = threading.Thread(target=task)
        thread.start()
        return thread
    return wrapped

# Celery tasks
@celery.task(bind=True)
def long_running_task(self, task_type):
    """Example of a long-running task."""
    total_steps = 10
    
    for i in range(total_steps):
        time.sleep(1)  # Simulate work
        self.update_state(
            state='PROGRESS',
            meta={'current': i + 1, 'total': total_steps}
        )
    
    return {'status': 'completed'}

@celery.task(bind=True, max_retries=3)
def process_webhook(self, webhook_data):
    """Process webhook data with retry support."""
    try:
        # Process webhook data
        result = process_webhook_data(webhook_data)
        return result
    except Exception as exc:
        self.retry(exc=exc, countdown=60)  # Retry after 1 minute

# RQ tasks
def send_email(to_address, subject, body):
    """Send email using RQ."""
    # Add email sending logic here
    time.sleep(2)  # Simulate email sending
    return f"Email sent to {to_address}"

def generate_report(report_type, parameters):
    """Generate report using RQ."""
    # Add report generation logic here
    time.sleep(5)  # Simulate report generation
    return f"Report {report_type} generated"

# Routes
@app.route('/task', methods=['POST'])
def create_task():
    """Create a new background task."""
    task_type = request.json.get('type', 'default')
    task = long_running_task.delay(task_type)
    return jsonify({'task_id': task.id}), 202

@app.route('/task/<task_id>')
def get_task_status(task_id):
    """Get task status."""
    task = long_running_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task pending...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'status': task.info.get('status', ''),
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1)
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    return jsonify(response)

@app.route('/email', methods=['POST'])
def queue_email():
    """Queue an email for sending."""
    data = request.json
    job = task_queue.enqueue(
        send_email,
        data['to'],
        data['subject'],
        data['body'],
        job_timeout='1h'
    )
    return jsonify({'job_id': job.id}), 202

@app.route('/report', methods=['POST'])
def queue_report():
    """Queue a report for generation."""
    data = request.json
    job = task_queue.enqueue(
        generate_report,
        data['type'],
        data['parameters'],
        job_timeout='2h'
    )
    return jsonify({'job_id': job.id}), 202

# Async route example
@app.route('/async-data')
async def get_async_data():
    """Example of async route."""
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.example.com/data') as response:
            data = await response.json()
            return jsonify(data)

# WebSocket support
@app.websocket('/ws')
async def ws():
    """WebSocket endpoint."""
    while True:
        data = await websocket.receive()
        await websocket.send(f"Echo: {data}")

# Background task with threading
@app.route('/background-task')
def start_background_task():
    """Start a background task using threading."""
    @async_task
    def background_job():
        # Simulate long-running job
        time.sleep(10)
        app.logger.info('Background task completed')
    
    background_job()
    return jsonify({'status': 'Background task started'})

# Task monitoring
@app.route('/monitor/celery')
def monitor_celery():
    """Monitor Celery tasks."""
    i = celery.control.inspect()
    return jsonify({
        'active': i.active(),
        'reserved': i.reserved(),
        'scheduled': i.scheduled()
    })

@app.route('/monitor/rq')
def monitor_rq():
    """Monitor RQ tasks."""
    return jsonify({
        'queued': len(task_queue),
        'failed': len(task_queue.failed_job_registry),
        'workers': len(task_queue.workers)
    })

# Error handling for tasks
@celery.task(bind=True, max_retries=3, default_retry_delay=60)
def task_with_error_handling(self):
    """Example of task with error handling."""
    try:
        # Task logic here
        raise Exception('Task failed')
    except Exception as exc:
        if self.request.retries < self.max_retries:
            self.retry(exc=exc)
        else:
            # Log final failure
            logging.error('Task failed permanently')
            raise

if __name__ == '__main__':
    app.run(debug=True) 