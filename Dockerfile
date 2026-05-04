FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_v4.py .
COPY sepsis_risk_model.json .
COPY prenova_cell_bacteria_logo.png .

EXPOSE 8003

CMD ["python", "app_v4.py"]
