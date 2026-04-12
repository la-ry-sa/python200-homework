import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- scikit-learn API ---

# Q1

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression() 
model.fit(years, salary)
test_years = np.array([4, 8]).reshape(-1, 1)
salary_predicted = model.predict(test_years)

print(f"Slope (coefficient): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Predicted salaries: {salary_predicted}")

# Q2

x = np.array([10, 20, 30, 40, 50])
print(x.shape)
x2d = x.reshape(-1, 1)
print(x2d.shape)

# We need to always reshape a 1D array to 2D so that the train data has sample and 
# feature dimensions. 

# Q3

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Number of samples in each cluster: {np.bincount(labels)}")
figsize=(10, 5)
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
plt.title("Student Clusters Found by K-Means")
plt.xlabel("Study Hours (synthetic scale)")
plt.ylabel("Exam Scores (synthetic scale)")
plt.savefig('outputs/kmeans_clusters.png')
plt.close()

#------Linear Regression----------

# Q1

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

figsize=(10, 5)
plt.scatter(age, cost, c=smoker, cmap='coolwarm')
plt.title('Medical Cost vs Age')
plt.xlabel('Age')
plt.ylabel('Annual Medical Cost')
plt.savefig('outputs/cost_vs_age.png')
plt.close()

#The plot shows a clear positive correlation between age and medical cost, with 
# smokers generally having higher costs than non-smokers at the same age. 
# This suggests that both age and smoking status are important factors 
# influencing medical costs.

# Q2

X = age.reshape(-1, 1)
y = cost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Q3

model = LinearRegression()
model.fit(X_train, y_train)
print(f"Slope (coefficient): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
y_pred = model.predict(X_test)

print(f"RMSE: {np.sqrt(np.mean((y_pred - y_test) ** 2))}")
print(f"R² on the test set: {model.score(X_test, y_test)}")

# The slope of this regression line is the average increase in medical cost with each year. 
# Slope of ~200 means that on average, each year the annual medical cost increases 
# by ~$200.

# Q4

# Now add smoker as a second feature and fit a new model.
X_full = np.column_stack([age, smoker])
#Split, fit, and print the test R². Compare it to the R² from Question 3 -- 
#does adding the smoker flag help? Print both coefficients:

X_train, X_test, y_train, y_test = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(X_train, y_train)

print(f"R² on the test set: {model_full.score(X_test, y_test)}")
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# The smoker coefficient of ~15000 means that, smokers would spend extra ~$15000 per year on
# medical costs on average, compared to non-smokers. Smoking has a significant impact 
# on medical costs, likely due to the increased health risks.

# Q5

# Using the two-feature model from Linear Regression Question 4, create this plot for 
# the test set. Add a diagonal reference line, a title "Predicted vs Actual", 
# labeled axes, and save to outputs/predicted_vs_actual.png.

y_pred_test = model_full.predict(X_test)
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_test, y_test, alpha=0.7)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel("Predicted Medical Costs")
plt.ylabel("Actual Medical Costs")
plt.title("Predicted vs Actual")
plt.savefig('outputs/predicted_vs_actual.png')
plt.close()