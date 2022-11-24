FROM python:3.9

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY models/LogisticRegression_model.pkl /model.pkl
COPY app.py features.py build_features.py tests.py make_request.py heart_cleveland_upload.csv /

WORKDIR .

ENV PATH_TO_MODEL="model.pkl"
CMD ["uvicorn", "app:app_", "--host", "0.0.0.0", "--port", "8000"]
