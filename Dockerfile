FROM python:3.12.4

WORKDIR /app
COPY model/requirements.txt .
RUN pip install -r requirements.txt


COPY data/train.csv /app/data/train.csv
COPY data/test.csv /app/data/test.csv

COPY model/app/app.py .
COPY model/app/titanic.py .
# COPY titanic_model.pkl .

EXPOSE 5000

RUN python titanic.py

ENTRYPOINT [ "python", "app.py" ]