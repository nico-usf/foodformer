FROM locustio/locust

RUN pip install datasets Pillow

WORKDIR /home/locust
COPY download_and_prepare_dataset.py .
RUN python download_and_prepare_dataset.py
COPY locustfile.py .

CMD ["-f", "/home/locust/locustfile.py"]
