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

# All predictions are correct - there's no confusion.

#-------------Decision Trees------------

# Q1

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
preds1 = tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds1))
print(classification_report(y_test, preds1))

#------------Logistic Regression------------

# Q1

# When troubleshooting the ValueError, AI suggested a different solver 
# that is able to handle multiclass

model_1 = LogisticRegression(C=0.01, max_iter=1000, solver='lbfgs')
model_1.fit(X_train_scaled, y_train)
print(model_1.C)
print(np.abs(model_1.coef_).sum())

model_2 = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
model_2.fit(X_train_scaled, y_train)
print(model_2.C)
print(np.abs(model_2.coef_).sum())

model_3 = LogisticRegression(C=100, max_iter=1000, solver='lbfgs')
model_3.fit(X_train_scaled, y_train)
print(model_3.C)
print(np.abs(model_3.coef_).sum())

# As C increases, total coefficient magnitude increases, while regularization decreases.

#----------PCA-----------

# Q1

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

print(X_digits.shape)
print(images.shape)

fig, axes = plt.subplots(1, 10, figsize=(10, 2))
for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(digit)
    axes[digit].axis('off')
plt.savefig("assignments_03/outputs/sample_digits.png")
plt.close()

# Q2

pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)
plt.figure(figsize=(6, 5))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Digits Dataset")
plt.tight_layout()
plt.colorbar(scatter, label='Digit')
plt.savefig("assignments_03/outputs/pca_2d_projection.png")
plt.close()

# Q3

plt.figure(figsize=(6, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("PCA Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.axhline(0.8)
plt.savefig("assignments_03/outputs/pca_variance_explained.png")
plt.close()

# We need ~12 components to explain 80% of variance.

# Q4

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

components = [2, 5, 15, 40]

fig, axes = plt.subplots(5,5)
for i in range(5):
    axes[0, 0].set_title("Original")
    axes[i, 0].imshow(images[i], cmap='gray_r')
    axes[i, 0].axis('off')
    for j, n in enumerate(components):    
        axes[0, j+1].set_title(f"n={n}")
        axes[i, j+1].imshow(reconstruct_digit(i, scores, pca, n), cmap='gray_r')
        axes[i, j+1].axis('off')    
plt.savefig("assignments_03/outputs/pca_reconstructions.png")
plt.close()

# Digits become recognizable at n=15 which matches the variance curve