FROM tensorflow/tensorflow:2.9.1

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt