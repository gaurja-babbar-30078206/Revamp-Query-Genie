FROM python:3.11.5-slim-bookworm

RUN apt-get update -y && \ 
    apt-get install binutils -y

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY app/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
