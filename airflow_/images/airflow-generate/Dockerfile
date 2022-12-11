FROM airflow-ml-base:latest

COPY heart_cleveland_upload.csv /heart_cleveland_upload.csv
COPY features.py /features.py
COPY generate_data.py /generate_data.py

ENTRYPOINT ["python", "generate_data.py"]
