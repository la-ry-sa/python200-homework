#----------------------Part 2: Project -- LLM Transform Pipeline--------------------

import json
import os
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

load_dotenv()

#----------------------Setup---------------------------

ACCOUNT_URL = "https://larysactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

#---------------------Step 1: Read---------------------

today = date.today().isoformat()
credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

blob_path = f"raw/{today}/weather.json"

try:
    raw = container.download_blob(blob_path).readall()
    data = json.loads(raw.decode("utf-8"))
except Exception:
    with open("assignments_10/resources/weather.json", "r", encoding="utf-8") as file:
        data = json.load(file)

hourly = data["hourly"]
records = []
for i in range(len(hourly["time"])):
    records.append({
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    })

print(f"Loaded {len(records)} records")

#----------------Step 2: Transform-------------------

VALID_LABELS = {"good", "marginal", "bad"}
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enriched = []
for i, record in enumerate(records[:24]):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ]
    )
    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in VALID_LABELS else "unknown"
    enriched.append({**record, "conditions": label})
    if (i + 1) % 6 == 0:
        print(f"  Processed {i + 1} records...")

#-----------------Step 3: Write------------------------

processed_path = f"processed/{today}/weather_classified.json"
payload = json.dumps(enriched).encode("utf-8")
container.upload_blob(processed_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {processed_path}")

#----------------Step 4: Spot-Check-----------------------

df = pd.DataFrame(enriched)
print(df["conditions"].value_counts())
print(f"\nFirst 5 rows:")
print(df.head())

#--------------Step 5: Save Output-------------------------

with open("assignments_10/outputs/first_10_records.json", "w", encoding="utf-8") as file:
    file.write(df.head(10).to_json(orient='records', indent=2))