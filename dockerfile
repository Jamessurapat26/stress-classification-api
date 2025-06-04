
FROM python:3.10-slim


WORKDIR /model-classification-api


COPY ./requirements.txt /model-classification-api/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /model-classification-api/requirements.txt


COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]