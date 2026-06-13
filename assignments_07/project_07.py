#--------------------Part 2: Mini-Project — World Happiness Agent----------------

from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

#--------------------Pre-task: Load the Data--------------------------

DATA_PATH = "assignments_01/outputs/merged_happiness.csv"

#--------------------Task 1: Define Your Tools--------------------

df = None

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.
    Store data from the csv located at the path to a global dataframe.
    Returns:
        A dict of "shape" and "columns".
    """
    global df
    df = pd.read_csv(DATA_PATH)
    myDict = {"shape": df.shape, "columns": df.columns.to_list()}
    return myDict

@tool
def summarize_column(column: str) -> dict:
    """Show basic summary statistics for a column (uses pandas.describe).
    Args:
        column: column name to summarize. If None, return an error.
    Returns:
        a dict or summary statistics
    """
    if (df is None or column not in df.columns):
            return {"error": f"No data loaded or column not found."}
    return df[column].describe().to_dict()

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    Args:
        col1: column of a dataframe created from csv.
        col2: column of a dataframe created from csv.
    Returns:
        a dict with "col1", "col2", "pearson_r", and "p_value" (rounded to 4 decimal places). 
    """
    if (df is None or col1 not in df.columns or col2 not in df.columns):
        return {"error": "No csv loaded or column doesn't exist."}
    r, p = pearsonr(df[col1], df[col2])
    return {"col1": col1, "col2": col2, "pearson_r": round(r, 4), "p_value": round(p, 4)}

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    Args:
        column: column of a dataframe created from csv.
        year: a specific year to filter the data. 
        n: a number of top countries to return, default value is 5.
    Returns: 
        a dict with "country", "column"
    """
    if (df is None or column not in df.columns or year not in df["Year"].values):
        return {"error": "No csv loaded or invalid year or column doesn't exist."}
    df_year = df[df["Year"] == year]
    df_sorted = df_year.sort_values(column, ascending = False)
    return df_sorted[["Country", column]].head(n).to_dict(orient="records")


#--------------Task 2: Build the Agent---------------------

model = OpenAIServerModel(api_key=os.environ["OPENAI_API_KEY"], model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
When using tools, always call tools in code blocks; when done, return a final answer and 
do not ask the user what to do next.
The dataset is stored internally by the tools. Do not assume or access a dataframe variable 
directly. Use the return value from load_happiness_data() for shape and columns.
Do not assume or access dataframe variables directly after tool calls. For custom plots or 
analyses not covered by tools, read the CSV directly from DATA_PATH using pandas.
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

if __name__ == "__main__":

#-----------------Task 3: Run Guided Queries--------------
    queries = [
    "Use only the load_happiness_data tool and return its result.",
    "Summarize the 'Happiness score' column.",
    "What is the correlation between 'GDP per capita' and 'Happiness score'? Is it statistically significant?",
    "Show me the top 5 happiest countries (by 'Happiness score' column) in 2020.",
    """Plot 'Happiness score' over the years as a line chart, with one line per 
    'Regional indicator'. For this plot, read the CSV directly from 
    assignments_01/outputs/merged_happiness.csv using pandas. Do not use 
    load_happiness_data() for the plot because that tool only returns shape 
    and columns. Group by Year and Regional indicator, compute mean happiness score, 
    plot with matplotlib. Save the plot to assignments_07/outputs/happiness_by_region.png.
    """,
]    
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

#-----------------Task 4: Your Own Questions----------------

response = agent.run(queries[0], reset=False)
# My query 1
my_query_1 = "What is the correlation between ‘Healthy life expectancy’ and ‘Happiness score’?"
response_1 = agent.run(my_query_1, reset=False)
print(response_1)
# Comment: Did this trigger tool use, code generation, or both?
# Triggered both and still did not succeed.

# My query 2
my_query_2 = "Plot average 'Generosity' by year"
response_2 = agent.run(my_query_2, reset=False)
print(response_2)
# Comment: Did this trigger tool use, code generation, or both?
# Triggered code generation. Again, unsuccessful.

#---------------Task 5: Reflection-------------------

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#
#   The agent added a comment about significance of the correlation. It used the p-value correctly.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#
#  The original plot query - the agent was unable to resolve it. I had to adjust both
# system prompt and query to be less vague and complete the task.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
#  Data cleaning - drop duplicates, replace NaN etc.