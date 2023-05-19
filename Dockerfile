FROM python:3

RUN pip install jupyter

WORKDIR /app/OS

COPY os.ipynb /app/OS

RUN jupyter nbconvert --to python os.ipynb

CMD ["python", "os.py"]


