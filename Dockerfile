FROM python:3.10.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./serving /code/serving
COPY ./models /code/models

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]