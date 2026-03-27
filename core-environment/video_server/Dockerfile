# ./video_server/Dockerfile
FROM python:3.9

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY / /app/

EXPOSE 5000

# Example: if your Flask app is in app.py or main.py
# CMD ["python", "run_server.py"]
CMD ["python", "run_server.py"]
# For production, consider Gunicorn or Waitress:
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]