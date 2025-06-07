# syntax=docker/dockerfile:1
FROM python:3.12.10-bookworm
ENV PYTHONUNBUFFERED=1
WORKDIR /code

# install requirements
COPY requirements.txt /code/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m spacy download pl_core_news_sm
