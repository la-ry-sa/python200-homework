from prefect import task, flow
import pandas as pd
import numpy as np

@task
def create_series(arr):
    values = pd.Series(arr)
    return values

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow
def data_pipeline(arr):
    values = create_series(arr)
    cleaned = clean_data(values)
    return summarize_data(cleaned)

if __name__ == "__main__":
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    data_pipeline(arr)