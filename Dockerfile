FROM python:3.9

WORKDIR /app

COPY /home/ubuntu/settings/settings.py /app/video_detector

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000

CMD python /app/manage.py runserver 0.0.0.0:8000
