FROM python:3.10.11-slim-buster

WORKDIR /app

RUN apt update -y && apt install ffmpeg -y

COPY requirements.txt /app/

RUN pip install --upgrade pip 

RUN pip install -r requirements.txt

COPY . /app/

EXPOSE 8000

CMD [ "uvicorn", "tags:app","--reload","--host","0.0.0.0","--port","8000" ]

