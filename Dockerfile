FROM tensorflow/tensorflow:2.9.1

# RUN apt-get -y update && apt-get install -y xvfb ffmpeg freeglut3-dev

RUN apt-get -y install python3-tk

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# ENTRYPOINT [ "python3", "main.py" ]