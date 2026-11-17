
FROM python:3.11-slim

WORKDIR /app

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Run the LLM evaluation
CMD ["python", "src/evaluation_runner.py"]
