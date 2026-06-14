#----------------------Prefect Orchestration---------------------

#----------------------Prefect Question 1------------------------

# @task is used for an isolated function that can be used in @flow.
# @flow calls the tasks. I wouldn't decorate the C to F conversion function with @task,
# because it's a separate action and not part of a flow.

#----------------------Prefect Question 2-------------------------

from prefect import task

@task(retries=3, retry_delay_seconds=30)
def call_api():
    pass #adding this line to avoid syntax error 

#----------------------Prefect Question 3-------------------------

# I would check Prefect logs to see what kind of error caused Transform to fail.

#---------------------Production Patterns-------------------------

#--------------------Production Question 1------------------------

# raise_for_status() marks the task as Failed, logs the error, stops the flow
# so the downstream tasks are not run and no bad data is saved.
# if response.status_code != 200: print("error") won't log the actual error 
# and won't stop the process - downstream tasks will run and write bad data.

#--------------------Production Question 2------------------------

# overwrite=True makes sure that corrupted data from the first run is replaced
# with the data from the successful run. Without it the second run would fail as
# the blob already exists.

#--------------------Production Question 3------------------------

from prefect.logging import get_run_logger

@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()

    logger.info(
        f"Loaded {len(records)} records for {blob_path}"
    )