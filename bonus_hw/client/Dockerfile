FROM python:3.9-slim

RUN pip install --no-cache-dir \
    tritonclient[http] \
    rich \
    faiss-cpu \
    numpy

WORKDIR /app
COPY tester.py .

CMD ["python", "tester.py"]