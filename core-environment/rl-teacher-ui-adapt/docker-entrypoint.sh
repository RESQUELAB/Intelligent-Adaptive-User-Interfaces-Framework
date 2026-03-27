#!/bin/sh

echo "Waiting for postgres..."
python << END
import socket
import time

host = "db"
port = 5432
while True:
    try:
        with socket.create_connection((host, port), timeout=1):
            break
    except OSError:
        print("Waiting for PostgreSQL...")
        time.sleep(0.5)
END

echo "PostgreSQL started"

# Apply database migrations
python human-feedback-api/manage.py makemigrations
python human-feedback-api/manage.py migrate
python human-feedback-api/manage.py collectstatic --noinput

echo "Starting Django server..."
python human-feedback-api/manage.py runserver 0.0.0.0:8000
