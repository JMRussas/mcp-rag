FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pipeline (build index): docker run mcp-rag python pipeline.py rebuild
# Server (MCP stdio):     default entrypoint
ENTRYPOINT ["python", "server.py"]
