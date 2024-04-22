FROM python:3.12.3
WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt

COPY . /code
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host 0.0.0.0", "