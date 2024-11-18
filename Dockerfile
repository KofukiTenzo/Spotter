FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt --timeout=30

COPY . /app

EXPOSE 8000

RUN python manage.py migrate

CMD python /app/manage.py runserver 0.0.0.0:8000
