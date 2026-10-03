FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app --uid 10001 app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY corpus /app/corpus
RUN pip install --no-cache-dir . && chown -R app:app /app
USER 10001
ENV RAG_CORPUS_DIR=/app/corpus
EXPOSE 8000
CMD ["uvicorn", "ragops.serving:app", "--host", "0.0.0.0", "--port", "8000"]
