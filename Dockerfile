ARG PY=3.14
FROM python:${PY}-slim
WORKDIR /app
RUN useradd --create-home appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/__init__.py src/__init__.py
COPY src/api.py src/api.py
COPY models models
USER appuser
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
