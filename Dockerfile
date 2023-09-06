FROM python:3.10.10

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY ./serving ./serving
COPY ./models ./models

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "80"]