from datetime import timedelta, datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "data_generation",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=datetime(2022, 12, 4),
) as dag:
    generate = DockerOperator(
        image="airflow-generate",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/vladislavgadzihanov/PycharmProjects/pythonProject/airflow_/data",
                      target="/data",
                      type='bind')]
    )
