FROM python:3.9

WORKDIR /app

COPY testcube.py .

RUN pip install -U pip && \
    pip install -U numpy

CMD [ "python", "./testcube.py" ]
