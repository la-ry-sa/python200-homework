import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

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

#---- Matplotlib ---
# Matplotlib Q1

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2

subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.ylim(0, 100)
plt.show()

# Matplotlib Q3

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Matplotlib Q4

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(x, y) 
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")
ax2.set_ylim(0, 100)

plt.tight_layout()  
plt.show()

#-----Descriptive Statistics-----
# Descriptive Statistics Q1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Standard Deviation: {np.std(data)}")

# Descriptive Statistics Q2

plt.figure(figsize=(8, 5))
plt.hist(np.random.normal(65, 10, 500), bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Descriptive Statistics Q3

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a, group_b], labels=['Group A', 'Group B'])
plt.title("Score Comparison")
plt.ylabel("Score")
plt.show()

# Descriptive Statistics Q4

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.boxplot([normal_data, skewed_data], labels=['Normal', 'Exponential'])
plt.title("Distribution Comparison")
plt.ylabel("Value")
plt.show()
# Exponential distribution is more skewed. 
# Median provides a more appropriate measure of central tendency for skewed data; 
# mean is more appropriate for normal distribution.

# Descriptive Statistics Q5

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
print(f"Data1 - Mean: {np.mean(data1)}, Median: {np.median(data1)}")
print(f"Data2 - Mean: {np.mean(data2)}, Median: {np.median(data2)}")
# Mean in data2 is significantly higher than median due to the outlier (150), 
# which skews the mean.

#-----Hypothesis Testing-----
# Hypothesis Testing Q1

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_val}")

# Hypothesis Testing Q2

if p_val < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")

# Hypothesis Testing Q3

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_val = stats.ttest_rel(before, after)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")

# Hypothesis Testing Q4

scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_val = stats.ttest_1samp(scores, 70)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")

# Hypothesis Testing Q5

stats.ttest_ind(group_a, group_b, alternative="less")
print(f"p-value: {p_val:.6f}")

# Hypothesis Testing Q6

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)

if p_val < 0.05:
    print("The test results suggest that Group B has significantly higher scores " \
    "than Group A, and this difference is unlikely to be due to chance.")

#-----Correlation------
# Correlation Q1

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
print(f"Correlation coefficient: {corr_matrix[0, 1]}")

# I expected correlation to be strong because y = 2x.  

# Correlation Q2

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r, p = pearsonr(x, y)
print(f"Correlation: {round(r, 2)}")
print(f"P-value: {round(p, 4)}")

# Correlation Q3

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
correlation_matrix = df.corr()
print(correlation_matrix)

# Correlation Q4

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Correlation Q5

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#------Pipelines------

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

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

def data_pipeline(arr):
    values = create_series(arr)
    cleaned = clean_data(values)
    return summarize_data(cleaned)

data_summary = data_pipeline(arr)
for key, value in data_summary.items():
    print(key, value)
