FROM python:3.10-slim-bullseye

WORKDIR /app

# install graphviz backend
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Not sure why, but you have to upgrade weaviate to v4. Installing fresh does not work
RUN pip install --pre -U "weaviate-client==4.*"

COPY . .

EXPOSE 8082:8082

# loads server.py module and the FastAPI `app` object
CMD uvicorn server:app --host 0.0.0.0 --port 8082
