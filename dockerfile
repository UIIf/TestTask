FROM python:3.11-slim

WORKDIR code

COPY requierments.txt .

RUN pip install --no-cache-dir -r requierments.txt

COPY ./code /code

EXPOSE 8000

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]