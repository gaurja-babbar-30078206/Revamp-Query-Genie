FROM python:3.11.5-slim-bookworm

RUN apt-get update -y && \ 
    apt-get install binutils -y

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install pdfplumber
RUN pip install tabulate

RUN python -m nltk.downloader punkt stopwords
COPY app/ .

ARG env
COPY ./${env}.env ./.env  

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
