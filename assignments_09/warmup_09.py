#--------------Azure Authentication------------------------------

#-------------Azure Authentication Question 1--------------------

# It relies on credentials from Azure CLI when running locally.
# The command "az login" must be run first to authenticate to Azure.
# DefaultAzureCredential tries several authentication methods in order.
# When it finds that Azure CLI is already logged in, it uses those credentials.

#--------------Azure Authentication Question 2----------------------

# az login still requires a user to enter login and password or do MFA, so it won't 
# work for a deployed pipeline. The pipeline uses a managed identity instead. 
# DefaultAzureCredential will automatically select a managed identity if it is 
# available. The code works unchanged because it still uses DefaultAzureCredential.

#-------------Azure Authentication Question 3------------------------

# 1. Attempted all authentication methods but none was successful. To diagnose I'd
# check error messages to see if az login was run locally or identity was found in Azure.

# 2. Found a credential but authentication failed. Would look for reasons and check
# the setup: expired login, identity configuration etc.

#-----------------------Blob Storage-------------------------

#----------------Blob Storage Question 1---------------------

# The lowest level is blob which is a file. Blobs are grouped in containers - 
# next level in hierarchy. Containers are stored in a storage account - the top level.
# Analogy: a neighborhood is a storage account, houses are containers, things stored in 
# each house are blobs.

#-----------------Blob Storage Question 2-----------------

# A REST API returns a JSON payload each hour. You need to store the raw responses for 
# reprocessing later.

# I'd use Blob Storage - the data is raw and just needs to be stored for processing 
# later.

# Your pipeline produces a table of 50 million customer transactions that your 
# analytics team queries by date range and customer ID every day.

# This is a scenario for Azure SQL, since transactions are structured data and 
# will need to be queried.

# A computer vision model produces image embeddings as NumPy arrays. You need to save 
# them between pipeline runs.

# Blob Storage - it would be hard to store image embeddings in a DB without processing.

#----------------Blob Storage Question 3---------------

from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
container_client = ContainerClient(
    account_url="https://larysactd2026sa.blob.core.windows.net",
    container_name="larysa7",
    credential=credential
)

def list_container(container_client):
    for blob in container_client.list_blobs():
        print(blob.name, blob.size)

#------------Blob Storage Question 4-------------------

def upload_text(container_client, blob_name, text):
    container_client.upload_blob(text.encode("utf-8"), blob_name=blob_name, overwrite=True)