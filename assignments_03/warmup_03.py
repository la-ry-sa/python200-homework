import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

#--------Preprocessing----------

# Q1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Q2

scaler_1 = StandardScaler()
X_train_scaled = scaler_1.fit_transform(X_train)
X_test_scaled = scaler_1.transform(X_test)

print(f"Mean of each column: {X_train_scaled.mean(axis=0)}")

# We only fit on train set to avoid 'showing' the test data to the model and prevent
# data leakage.

#-------KNN--------

# Q1

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Q2

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
preds = knn.predict(X_test_scaled)

print("Accuracy for scaled data:", accuracy_score(y_test, preds))

# Scaling reduced performance slightly. This might happen because 

# Q3

cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print(cv_scores)
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")

# Q4

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")

# I'd choose k=7 - highest mean and lowest std

#--------Classifier Evaluation----------

# Q1

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

cm = confusion_matrix(y_test, preds)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=iris.target_names)
disp.plot(colorbar=False)
plt.title("Confusion Matrix")
plt.savefig("assignments_03/outputs/knn_confusion_matrix.png")
plt.close()

# All predictions are correct - there's no confusion