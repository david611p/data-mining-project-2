import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
import time
 
df = pd.read_csv('student-mat.csv', sep=';')
df['passed'] = (df['G3'] >= 10).astype(int)
df = df.drop(columns=['G3'])

train_set, test_set = train_test_split(df, test_size=0.10, random_state=42)

y_train = train_set['passed']
X_train = train_set.drop(columns=['passed'])
y_test = test_set['passed']
X_test = test_set.drop(columns=['passed'])

categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)

pca = PCA(n_components=29)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_processed = preprocessor.transform(X_test)
X_test_pca = pca.transform(X_test_processed)

rf_params = [50, 100, 200]
svm_params = [0.1, 1, 10]

rf_results = []
svm_results = []

cv = KFold(n_splits=5, shuffle=True, random_state=42)

for n in rf_params:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    scores = cross_val_score(model, X_train_pca, y_train, cv=cv, scoring='accuracy')
    for score in scores:
        rf_results.append({'n_estimators': str(n), 'Accuracy': score})

for c in svm_params:
    model = SVC(C=c, random_state=42)
    scores = cross_val_score(model, X_train_pca, y_train, cv=cv, scoring='accuracy')
    for score in scores:
        svm_results.append({'C': str(c), 'Accuracy': score})

rf_df = pd.DataFrame(rf_results)
svm_df = pd.DataFrame(svm_results)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(x='n_estimators', y='Accuracy', data=rf_df, ax=axes[0])
axes[0].set_title('Random Forest CV Results (Mjob with PCA)')
axes[0].set_xlabel('n_estimators')
axes[0].set_ylabel('Accuracy')

sns.barplot(x='C', y='Accuracy', data=svm_df, ax=axes[1])
axes[1].set_title('SVM CV Results (Mjob with PCA)')
axes[1].set_xlabel('C (Regularization)')
axes[1].set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig('cv_results_mjob_pca.png')



print("Random Forest Results (PCA Pass/Fail, n_estimators=100):")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
start_train = time.time()
rf_model.fit(X_train_pca, y_train)
end_train = time.time()
train_time = end_train - start_train
train_acc = accuracy_score(y_train, rf_model.predict(X_train_pca))
print(f"Training Time: {train_time:.4f}s")
print(f"Training Accuracy: {train_acc:.4f}")

start_test = time.time()
y_pred_rf = rf_model.predict(X_test_pca)
end_test = time.time()
test_time = end_test - start_test
test_acc = accuracy_score(y_test, y_pred_rf)
print(f"Testing Time: {test_time:.4f}s")
print(f"Testing Accuracy: {test_acc:.4f}")

print("\nSVM Results (PCA Pass/Fail, C=1):")
svm_model = SVC(C=1, random_state=42)
start_train = time.time()
svm_model.fit(X_train_pca, y_train)
end_train = time.time()
train_time = end_train - start_train
train_acc = accuracy_score(y_train, svm_model.predict(X_train_pca))
print(f"Training Time: {train_time:.4f}s")
print(f"Training Accuracy: {train_acc:.4f}")

start_test = time.time()
y_pred_svm = svm_model.predict(X_test_pca)
end_test = time.time()
test_time = end_test - start_test
test_acc = accuracy_score(y_test, y_pred_svm)
print(f"Testing Time: {test_time:.4f}s")
print(f"Testing Accuracy: {test_acc:.4f}")

