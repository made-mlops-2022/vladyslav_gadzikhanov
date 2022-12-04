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
        "pipeline",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=datetime(2022, 12, 4),
) as dag:

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="/data/raw/{{ ds }} /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        auto_remove="always",
        mounts=[Mount(source="/Users/vladislavgadzihanov/PycharmProjects/pythonProject/airflow_/data",
                      target="/data",
                      type='bind')]
    )

    fit = DockerOperator(
        image="airflow-fit",
        command="/data/processed/{{ ds }} /data/models/{{ ds }}",
        task_id="docker-airflow-fit",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/vladislavgadzihanov/PycharmProjects/pythonProject/airflow_/data",
                      target="/data",
                      type='bind')]
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="/data/models/{{ ds }} /data/processed/{{ ds }} /data/predictions/{{ ds }}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/vladislavgadzihanov/PycharmProjects/pythonProject/airflow_/data",
                      target="/data",
                      type='bind')]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="/data/processed/{{ ds }} /data/predictions/{{ ds }} /data/models/{{ ds }}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/Users/vladislavgadzihanov/PycharmProjects/pythonProject/airflow_/data",
                      target="/data",
                      type='bind')]
    )

    preprocess >> fit >> predict >> validate
