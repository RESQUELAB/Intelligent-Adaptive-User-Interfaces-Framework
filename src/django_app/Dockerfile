# ./django_app/Dockerfile
FROM python:3.6

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY source_code/ .
COPY source_code/docker-entrypoint.sh docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir python-dotenv
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir -e .
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir -e ./human-feedback-api
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir pandas

RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir -e agents/ga3c
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir -e ui_adapt


EXPOSE 8000

# The entrypoint script will run migrations and then start the server
# CMD ["ls","-la"]
ENTRYPOINT ["./docker-entrypoint.sh"]
