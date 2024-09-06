# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load data from CSV (simulated in this case as CSV was provided as text)
data = pd.read_csv('VocAbsDataTps.csv')

# Split features (x) and target (y)
x = data.drop(columns=['label'])
y = data['label']
print(x)

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Apply PCA for feature reduction (we try 5 components)
pca = PCA(n_components=5)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Train Decision Tree Classifier
clf = svm.SVC()
clf.fit(x_train_pca, y_train)

# Predictions
y_pred = clf.predict(x_test_pca)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification report and confusion matrix
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Output results
accuracy, class_report
print('Accuracy: ', accuracy)
print('Classification Report: \n', class_report)

# add the classification report in excel file

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
#report_df.to_csv('SVM_classification_report.csv', sep='\t', index=True)

with pd.ExcelWriter('classification_reportSVM.xlsx', engine='xlsxwriter') as writer:
    report_df.to_excel(writer, sheet_name='SVM', index=True)


# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.color_palette("Blues")
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Support Vector Machine Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('CMatrixSVM.png')

# Plot PCA explained variance
plt.figure(figsize=(6,4))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, color='green')
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')
plt.savefig('PCA_SVM.png')



# Plot 10 random samples from the dataset (assuming each row represents a sample)
gases = data['label'].values
data_values = data.drop(columns=['label']).values  # Drop the label column to get the features

num_samples, num_wavelengths = data_values.shape

# Plot 5 random samples
plt.figure(figsize=(10, 6))
for i in range(5):
    random_sample = np.random.randint(0, num_samples)
    plt.plot(range(1, num_wavelengths + 1), data_values[random_sample, :], label=f'Sample {random_sample} - {gases[random_sample]}')


plt.title('10 Random FTIR Samples')
plt.xlabel('Wavelength (Feature Index)')
plt.ylabel('Absorbance')



# Show both plots
plt.tight_layout()
plt.show()


