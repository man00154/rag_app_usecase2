FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential cmake libopenblas-dev libomp-dev

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
