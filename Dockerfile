# syntax=docker/dockerfile:1
FROM python:3.12.10-bookworm
ENV PYTHONUNBUFFERED=1
WORKDIR /code

# install requirements
COPY requirements.txt /code/
COPY import_stanza_pl.py /code/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m spacy download pl_core_news_sm
RUN python import_stanza_pl.py

EXPOSE 8001
CMD ["uvicorn", "app.main:my_app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
