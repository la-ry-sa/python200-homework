import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#--------Task 1: Load and Explore--------

# Load data with the correct separator, print first 5 rows, shape and data types.

df = pd.read_csv('assignments_02/student_performance_math.csv', sep=';')
print(df.head(5))
print(df.shape)
print(df.dtypes)

# Histogram of G3 grades, save as g3_distribution.png.

plt.figure(figsize=(10,6))
plt.hist(df['G3'], bins=21, color='skyblue', edgecolor='black')
plt.title("Distribution of Final Math Grades")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.savefig('assignments_02/outputs/g3_distribution.png')
plt.close()

#-------Task 2: Preprocess the Data-------

# Separate rows where G3 = 0 to a separate dataframe; print first 5 rows.
df_cleaned = df[df['G3'] != 0]
print(df_cleaned.head())

# Compare shapes of both dataframes to confirm rows with G3 = 0 were removed. 
# Keeping the rows would distort the model, because those students didn't take the exam.
print(df.shape)
print(df_cleaned.shape)
print(f"Number of rows removed: {df.shape[0] - df_cleaned.shape[0]}")

df_cleaned['sex'] = df_cleaned['sex'].map({'F': '0', 'M': '1'})
df_cleaned.replace({'yes': 1, 'no': 0}, inplace=True)
df_cleaned = df_cleaned.astype(int)
print(df_cleaned.dtypes)

correlation1 = df['absences'].corr(df['G3'])
print(f"Correlation (with G3=0): {correlation1}")
correlation2 = df_cleaned['absences'].corr(df_cleaned['G3'])
print(f"Correlation (without G3=0): {correlation2}")
      
# Correlation is much stronger in the filtered dataframe, because the rows with G3=0 
# were likely students who didn't take the exam, which would distort the relationship 
# between absences and final grades.
# explore scatter plots to help understand this.

plt.figure(figsize=(10,6))
plt.subplot(1, 2, 1)
plt.scatter(df['absences'], df['G3'], alpha=0.5)
plt.title("Absences vs Final Grade (with G3=0)")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")
plt.subplot(1, 2, 2)
plt.scatter(df_cleaned['absences'], df_cleaned['G3'], alpha=0.5, color='orange')
plt.title("Absences vs Final Grade (without G3=0)")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig('assignments_02/outputs/absences_vs_grade.png')
plt.close()

#-----------Task 3: Exploratory Data Analysis-------

correlations_g3 = df_cleaned.drop(columns=['G3']).corrwith(df_cleaned['G3'])
print(correlations_g3)
corr_ranked = correlations_g3.sort_values(ascending=False)
print(corr_ranked)

# G2 and G1 have the strongest correlation with G3, which is expected.
# A bit surprising is that mother's education has a stronger correlation 
# with G3 than father's education. It was also a bit unexpected that age 
# has a negative correlation.

# Scatter plot of absences vs G3.
plt.figure(figsize=(10,6))
plt.scatter(df_cleaned['absences'], df_cleaned['G3'], color='green')
plt.title('Absences vs. Final Grades')
plt.xlabel('Absences')
plt.ylabel('Final Grades')
plt.savefig('assignments_02/outputs/absences_vs_g3.png')
plt.close()

# The scatter plot shows a negative relationship between absences and final grades.

# Box plot of G3 grades by mother's education level.
plt.figure(figsize=(10,6))
df_cleaned.boxplot(column='G3', by='Medu', grid=False)
plt.title('Final Grades by Mother\'s Education Level')
plt.suptitle('')
plt.xlabel('Mother\'s Education Level')
plt.ylabel('Final Grades (G3)')
plt.savefig('assignments_02/outputs/g3_by_mother_education.png')
plt.close()

# The box plot shows that students whose mothers have higher education levels 
# tend to have higher final grades.

#-------Task 4: Baseline Model-------

X = df_cleaned[['failures']]
y = df_cleaned['G3']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Slope:", model.coef_[0])
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

# Slope of about -1.4 means that with each failure final grade decreases by
# about 1.4 points. RMSE of ~2.97 means that on average, the model's predictions 
# are off by about 3 points.
# R² of ~0.09 is lower than I expected and indicates that number of failures is 
# not a strong predictor of final grades.   

#-------Task 5: Build the Full Model-------

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_cleaned[feature_cols].values
y = df_cleaned["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Slope:", model.coef_[0])
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")