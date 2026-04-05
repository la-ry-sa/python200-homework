import pandas as pd
import numpy as np

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

#--- NumPy ---
# NumPy Q1

arr = np.array([10, 20, 30, 40, 50])
print(f"Array shape: {arr.shape}")
print(f"Array data type: {arr.dtype}")
print(f"Array ndim: {arr.ndim}")

# NumPy Q2

arr2 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"Array shape: {arr2.shape}")
print(f"Array size: {arr2.size}")

# NumPy Q3

print(f"Sliced array: \n{arr2[0:2, 0:2]}")

#NumPy Q4

print(f"2D array of zeros: \n{np.zeros((3, 4))}")
print(f"2D array of ones: \n{np.ones((2, 5))}")

# NumPy Q5

arr3 = np.arange(0, 50, 5)
print(f"Array: \n{arr3}")
print(f"Array shape: {arr3.shape}")
print(f"Array mean: {arr3.mean()}")
print(f"Array sum: {arr3.sum()}")
print(f"Array standard deviation: {arr3.std()}")

# NumPy Q6

arr4 = np.random.normal(0, 1, 200)
print(f"Array mean: {arr4.mean()}")
print(f"Array standard deviation: {arr4.std()}")