import pandas as pd
import numpy as np
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset from CSV
data = pd.read_csv('VocAbsDataTps.csv')

# Split features (x) and target (y)
x = data.drop(columns=['label'])
y = data['label']

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Apply PCA for feature reduction (we reduce to 5 components here for simplicity)
pca = PCA(n_components=5)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)


# Train Gradient Boosting Classifier
gbt_clf = GradientBoostingClassifier(random_state=42)
gbt_clf.fit(x_train_pca, y_train)

# Predictions from RF and GBT
y_pred_gbt = gbt_clf.predict(x_test_pca)

# Accuracy
accuracy_gbt = accuracy_score(y_test, y_pred_gbt)

# Classification report
class_report_gbt = classification_report(y_test, y_pred_gbt)

# Confusion matrices
conf_matrix_gbt = confusion_matrix(y_test, y_pred_gbt)

report_dict = classification_report(y_test, y_pred_gbt, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
#report_df.to_csv('DT_classification_report.csv', sep='\t', index=True)

with pd.ExcelWriter('classification_reportGBT.xlsx', engine='xlsxwriter') as writer:
    report_df.to_excel(writer, sheet_name='GBT', index=True)


# Output results
print(f"Gradient Boosting Accuracy: {accuracy_gbt:.4f}")
print("\nGradient Boosting Classification Report:")
print(class_report_gbt)

# Plot confusion matrices for both models
plt.figure(figsize=(6,4))

# Gradient Boosting Confusion Matrix
sns.heatmap(conf_matrix_gbt, annot=True, fmt='d', cmap='Greens', xticklabels=gbt_clf.classes_, yticklabels=gbt_clf.classes_)
plt.title('Gradient Boosting Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('CMatrixGBT.png')

# Plot PCA explained variance
plt.figure(figsize=(6,4))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, color='green')
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')
plt.savefig('PCA_GBT.png')

# Show both confusion matrix plots
plt.tight_layout()
plt.show()


