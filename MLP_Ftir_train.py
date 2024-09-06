import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Train Random Forest Classifier
rf_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
rf_clf.fit(x_train_pca, y_train)

# Predictions from RF and GBT
y_pred_rf = rf_clf.predict(x_test_pca)


# Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)


# Classification report
class_report_rf = classification_report(y_test, y_pred_rf)


# Confusion matrices
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

report_dict = classification_report(y_test, y_pred_rf, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
#report_df.to_csv('DT_classification_report.csv', sep='\t', index=True)

with pd.ExcelWriter('classification_reportMLP.xlsx', engine='xlsxwriter') as writer:
    report_df.to_excel(writer, sheet_name='MLP', index=True)


# Output results
print(f"MLP Accuracy: {accuracy_rf:.4f}")
print("\nMLP Classification Report:")
print(class_report_rf)

# Plot confusion matrices for both models
plt.figure(figsize=(6,4))

# Random Forest Confusion Matrix
sns.color_palette("YlOrBr")
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='YlOrBr', xticklabels=rf_clf.classes_, yticklabels=rf_clf.classes_)
plt.title('MLP Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('CMatrixMLP.png')

# Plot PCA explained variance
plt.figure(figsize=(6,4))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, color='green')
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')
plt.savefig('PCA_MLP.png')


# Show both confusion matrix plots
plt.tight_layout()
plt.show()



