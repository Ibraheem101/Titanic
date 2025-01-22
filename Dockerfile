FROM python:3.12.4

WORKDIR /app
COPY model/requirements.txt .
# COPY model/app/ingest.py .
RUN pip install -r requirements.txt

COPY data/train.csv data/train.csv
COPY data/test.csv data/test.csv

COPY model/app/app.py .
COPY model/app/titanic.py .

EXPOSE 5000

ENTRYPOINT [ "sh", "-c", "python titanic.py && python app.py" ]