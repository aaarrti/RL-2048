FROM tensorflow/tensorflow:2.9.1

RUN apt-get -y update && apt-get install -y xvfb ffmpeg freeglut3-dev

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python3", "main.py" ]