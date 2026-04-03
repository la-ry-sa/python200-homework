import pandas as pd

# --- Pandas ---
# Pandas Q1

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f"Three first rows: {(df.head(3))}")
print(f"DF shape: {df.shape}")
print(f"Data Types: {df.dtypes[0:3]}")