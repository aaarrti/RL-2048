FROM tensorflow/tensorflow:2.9.1

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt