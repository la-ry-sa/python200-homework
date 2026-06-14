#--------------Project -- Full ETL Pipeline---------------------

import json
import os
import requests
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from prefect import task, flow

load_dotenv()

#-------------------Extract task--------------------

ACCOUNT_URL = "https://larysactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
MAX_RECORDS = 24

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

@task(retries=2, retry_delay_seconds=10)
def extract_weather(latitude: float, longitude: float):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,precipitation"
        f"&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()

    print(f"Extracted forecast data for ({latitude}, {longitude})")
    return response.json()

#------------------------Transform task----------------------

@task
def transform(data: dict, max_records: int) -> list:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    hourly = data["hourly"]

    records = []
    for i in range(len(hourly["time"])):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })

    enriched = []

    for i, record in enumerate(records):
        if i < max_records:
            user_msg = (
                f"Temperature: {record['temperature_2m']}C, "
                f"Precipitation: {record['precipitation']}mm"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
            )

            raw_label = response.choices[0].message.content.strip().lower()
            label = raw_label if raw_label in VALID_LABELS else "unknown"

            if (i + 1) % 6 == 0:
                print(f"  Classified {i + 1}/{max_records} records")
        else: 
            label = 'not_classified'

        enriched.append({**record, "running_condition": label})

    print(f"Transform complete: {len(enriched)} records enriched")
    return enriched

#-----------------Load task-------------------

@task
def load(records: list, blob_path: str) -> None:
    credential = DefaultAzureCredential()

    container = ContainerClient(
        ACCOUNT_URL,
        CONTAINER,
        credential=credential
    )

    payload = json.dumps(records).encode("utf-8")

    container.upload_blob(
        blob_path,
        payload,
        overwrite=True
    )

    print(f"Loaded {len(payload)} bytes to {blob_path}")

#-----------------------Flow-------------------------

@flow(log_prints=True)
def etl_pipeline(
    latitude = 47.674,
    longitude = -122.122
):
    today = date.today().isoformat()

    blob_path = f"final/{today}/weather_etl.json"

    data = extract_weather(latitude, longitude)

    enriched = transform(
        data,
        max_records=MAX_RECORDS
    )

    load(enriched, blob_path)

    print(f"Pipeline complete. Results at {blob_path}")

if __name__ == "__main__":
    etl_pipeline()