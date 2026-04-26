import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

#--------Task 1: Load and Explore----------

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.head())

plt.figure(figsize=(10,6))
df.boxplot(column='word_freq_free', by='spam_label', grid = False)
plt.suptitle('')
plt.title('Word Frequency: "Free" by Email Type')
plt.xlabel('Email Type')
plt.xticks([1, 2], ['Ham', 'Spam'])
plt.ylabel('Frequency of "free"')
plt.savefig("assignments_03/outputs/free_frequency.png")
plt.close()

# Most emails don't have the word 'free'. Some spam emails still use the word frequently.

plt.figure(figsize=(10,6))
df.boxplot(column='char_freq_!', by='spam_label', grid = False)
plt.suptitle('')
plt.title('Character Frequency: "!" by Email Type')
plt.xlabel('Email Type')
plt.xticks([1, 2], ['Ham', 'Spam'])
plt.ylabel('Frequency of "!"')
plt.savefig("assignments_03/outputs/frequency_of_!.png")
plt.close()

# Similarly to 'free', most emails don't contain '!'. But spam emails tend to use it
# more frequently.

plt.figure(figsize=(10,6))
df.boxplot(column='capital_run_length_total', by='spam_label', grid = False)
plt.suptitle('')
plt.title('Frequency: "Longest Capital Letters Run" by Email Type')
plt.xlabel('Email Type')
plt.xticks([1, 2], ['Ham', 'Spam'])
plt.ylabel('Longest Capital Letters Run')
plt.savefig("assignments_03/outputs/longest_capital_run.png")
plt.close()

# Spam emails have longer capital runs.
# Many values are zero because most emails don’t contain certain words or 
# characters, so these features are sparse but still useful when they appear.
# The scales vary because some features are proportions (small values) and 
# others are counts (large values).
# This matters because features with larger values can have more influence 
# on some models, so scaling may be needed.

#------------Task 2: Prepare Your Data-----------
#---------------PCA preprocessing----------------

X = df.drop("spam_label", axis=1)
y = df["spam_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA()
pca.fit(X_train_scaled)
exp_var_cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(6, 5))
plt.plot(exp_var_cumsum)
plt.title("PCA Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.axhline(0.9)
plt.savefig("assignments_03/outputs/pca_variance.png")
plt.close()
n = 40 # based on the plot
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]

#------------Task 3: A Classifier Comparison-----------------

from sklearn.neighbors import KNeighborsClassifier
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
preds_unscaled = knn_unscaled.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds_unscaled))
print(classification_report(y_test, preds_unscaled))

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
preds_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, preds_scaled))
print(classification_report(y_test, preds_scaled))

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
preds_pca = knn_pca.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, preds_pca))
print(classification_report(y_test, preds_pca))

from sklearn.tree import DecisionTreeClassifier
for depth in [3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    preds_test = tree.predict(X_test)
    preds_train = tree.predict(X_train)
    print("Testing Accuracy:", accuracy_score(y_test, preds_test))
    print("Training Accuracy:", accuracy_score(y_train, preds_train))

# Depth 10 shows optimal balance between train and test accuracy.

prod_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
prod_tree.fit(X_train, y_train)
preds_test = prod_tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds_test))
print(classification_report(y_test, preds_test))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

model_1 = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
model_1.fit(X_train_scaled, y_train)
pred_1 = model_1.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, pred_1))

model_2 = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
model_2.fit(X_train_pca, y_train)
pred_2 = model_2.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, pred_2))

cm = confusion_matrix(y_test, pred_rf)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(colorbar=False)
plt.title("Confusion Matrix")
plt.savefig("assignments_03/outputs/best_model_confusion_matrix.png")
plt.close()

feature_imp_rf = rf.feature_importances_
feature_imp_tree = prod_tree.feature_importances_

pairs_rf = []
for name, importance in zip(X.columns, feature_imp_rf):
    pairs_rf.append((name, importance))
pairs_rf.sort(key=lambda pair: pair[1], reverse=True)
print(pairs_rf[:10])

pairs_tree = []
for name, importance in zip(X.columns, feature_imp_tree):
    pairs_tree.append((name, importance))
pairs_tree.sort(key=lambda pair: pair[1], reverse=True)
print(pairs_tree[:10])

# There's some difference between the two models. Tree model has words 'George' and
# 'our' which don't seem to be typical for spam emails.

top = pairs_rf[:10]
names = [pair[0] for pair in top]
values = [pair[1] for pair in top]
plt.figure(figsize=(10,6))
plt.bar(names, values)
plt.xticks(rotation=45)
plt.title("Feature Importances for RF Model")
plt.xlabel("Feature Name")
plt.ylabel("Importance")
plt.savefig("assignments_03/outputs/feature_importances.png")
plt.close()

# RF model is mostly consistent with my intuition - expected exclamation marks, 
# long capital runs, money, free etc. 'Remove' was somewhat unexpected though.

#----------Task 4: Cross-Validation---------------

cv_knn_unscaled = cross_val_score(knn_unscaled, X_train, y_train, cv=5)
print("KNN unscaled")

print(f"Mean: {cv_knn_unscaled.mean():.3f}")
print(f"Std:  {cv_knn_unscaled.std():.3f}")

cv_knn_scaled = cross_val_score(knn_scaled, X_train_scaled, y_train, cv=5)
print("KNN scaled")

print(f"Mean: {cv_knn_scaled.mean():.3f}")
print(f"Std:  {cv_knn_scaled.std():.3f}")

cv_knn_pca = cross_val_score(knn_pca, X_train_pca, y_train, cv=5)
print("KNN PCA")

print(f"Mean: {cv_knn_pca.mean():.3f}")
print(f"Std:  {cv_knn_pca.std():.3f}")

cv_prod_tree = cross_val_score(prod_tree, X_train, y_train, cv=5)
print("Tree Model")

print(f"Mean: {cv_prod_tree.mean():.3f}")
print(f"Std:  {cv_prod_tree.std():.3f}")

cv_rf = cross_val_score(rf, X_train, y_train, cv=5)
print("Random Forest Model")

print(f"Mean: {cv_rf.mean():.3f}")
print(f"Std:  {cv_rf.std():.3f}")

cv_model_1 = cross_val_score(model_1, X_train_scaled, y_train, cv=5)
print("Logistic Regression Scaled")

print(f"Mean: {cv_model_1.mean():.3f}")
print(f"Std:  {cv_model_1.std():.3f}")

cv_model_2 = cross_val_score(model_2, X_train_pca, y_train, cv=5)
print("Logistic Regression PCA")

print(f"Mean: {cv_model_2.mean():.3f}")
print(f"Std:  {cv_model_2.std():.3f}")

# RF model is the most accurate with the mean of 0.954.
# Logistics Regressin with PCA model is the most stable with std of 0.004.
# Ranking is consistent with the single train/test split.

#--------------Task 5: Building a Prediction Pipeline------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_pipeline_pred = rf_pipeline.predict(X_test)
print(classification_report(y_test, rf_pipeline_pred))

best_non_tree = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=40)),
    ("model", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

best_non_tree.fit(X_train, y_train)
non_tree_pred = best_non_tree.predict(X_test)
print(classification_report(y_test, non_tree_pred))

# The pipelines do not have the same structure. The tree-based pipeline includes 
# only the Random Forest model, while the non-tree pipeline also includes scaling 
# and PCA because those models require preprocessing. Pipelines are useful because 
# they ensure consistent data processing, reduce errors, and make models easier to 
# reuse and deploy.