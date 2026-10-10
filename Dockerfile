# Production image for Render / Railway / any Docker host
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py hsn_master.csv ./
COPY .streamlit ./.streamlit

ENV PORT=8501
EXPOSE 8501
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
