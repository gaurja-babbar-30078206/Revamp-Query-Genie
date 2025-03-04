FROM python:3.11.5-slim-bookworm

RUN apt-get update -y && \ 
    apt-get install binutils -y

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY src/. .

EXPOSE 8501
# CMD ["python", "app.py"]
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]