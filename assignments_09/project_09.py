from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from datetime import date
import json
import requests
import io
import pandas as pd
import warmup_09

#-----------------------SETUP------------------

ACCOUNT_URL = "https://larysactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

#------------------------Step 1: Extract----------------

# 47.674
# -122.122

url = "https://api.open-meteo.com/v1/forecast?latitude=47.674&longitude=-122.122&hourly=temperature_2m,precipitation&forecast_days=7"

response = requests.get(url)
response.raise_for_status()
data = response.json()

#-----------------------Step 2: Serialize--------------------

payload = json.dumps(data).encode("utf-8")

#----------------------Step 3: Load------------------------

today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url=ACCOUNT_URL,
    container_name=CONTAINER,
    credential=credential
)

container.upload_blob(blob_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {blob_path}")

#----------------------Step 4: Verify--------------------

warmup_09.list_container(container)

#----------------------Step 5: Read Back-------------------

raw = container.download_blob(blob_path).readall()
df = pd.DataFrame(json.loads(raw.decode("utf-8"))["hourly"])
print(f"\nFirst 5 rows:")
print(df.head())

with open("assignments_09/outputs/weather_raw.json", "w", encoding="utf-8") as file:
    file.write(raw.decode("utf-8"))