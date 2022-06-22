FROM tensorflow/tensorflow:2.9.1

WORKDIR /app

RUN apt-get update && apt-get install -y xvfb ffmpeg freeglut3-dev

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

#COPY . .

#ENTRYPOINT ["python3", "main.py"]