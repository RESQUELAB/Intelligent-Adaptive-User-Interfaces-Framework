# ./rlhf_server_1/Dockerfile
FROM python:3.6
# FROM tensorflow/tensorflow:1.10.0-py3
# RUN apt-get update && apt-get install -y build-essential
# RUN pip install tensorflow==1.10.0

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY source_code/ .
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir -e .
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir -e agents/ga3c
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir -e ./human-feedback-api
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir -e ui_adapt
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir django==1.11
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --no-cache-dir websocket-client==1.3.1 websocket-server==0.6.4
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-cache-dir psutil

EXPOSE 9998

# CMD ["ls" , "rl_teacher", "-la"]
# CMD ["cat", "setup.py"]
CMD ["python", "rl_teacher/teach.py", "-e", "UIAdaptation-v0", "-n", "cso", "-p", "human", "-L", "10", "-w", "1", "-tep", "1000000", "-m", "server", "-c", "4", "-V", "--port", "9997", "-i", "1000000"]
