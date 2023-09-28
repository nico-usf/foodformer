FROM locustio/locust

RUN pip install datasets
RUN python download_and_prepare_dataset.py
COPY /path/to/your/local/dataset /dataset
COPY locustfile.py /locustfile.py

CMD ["-f", "/locustfile.py"]
