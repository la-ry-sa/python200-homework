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

print(f"Three first rows: \n{df.head(3)}")
print(f"DF shape: {df.shape}")
print(f"Data Types: \n{df.dtypes[0:3]}")

# Pandas Q2

print(f"Passed with grade above 80: \n{(df[(df['grade'] > 80) & (df['passed'] == True)])}")

# Pandas Q3

df['grade_curved'] = df['grade'] + 5
print(f"DataFrame with curved grades: \n{df}")

# Pandas Q4

df['name_upper'] = df['name'].apply(str.upper)
print(f"Two columns together: \n{df[['name','name_upper']]}")

# Pandas Q5

print(f"Average grade by city: \n{df.groupby('city')['grade'].mean()}")

# Pandas Q6

df['city'] = df['city'].replace('Austin', 'Houston')
print(f"Updated city names: \n{df[['name','city']]}")

# Pandas Q7

df_sorted = df.sort_values('grade', ascending = False)
print(f"Top 3 students: \n{df_sorted.head(3)}")