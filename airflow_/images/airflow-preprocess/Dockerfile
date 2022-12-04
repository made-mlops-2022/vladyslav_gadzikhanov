FROM airflow-ml-base:latest

COPY features.py /features.py
COPY build_features.py /build_features.py
COPY preprocess.py /preprocess.py

ENTRYPOINT ["python", "preprocess.py"]