"""
@title: Naive Bayes Classifier for Covid outcome prediction
@author: Ahmed Bakr
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay

# Print header
print("\033[34m################################################################################\033[0m")
print("\033[34m--------------------------------------------------------------------------------\033[0m")
print("\t\t\tCovid Outcome Prediction using Naive Bayes")
print("\033[34m--------------------------------------------------------------------------------\033[0m")
print("\033[34m################################################################################\033[0m")

# Load the dataset
dataset = pd.read_csv('data.csv')

x = dataset.iloc[:, 1:14]
y = dataset.iloc[:, 14]

# Split the dataset into training, validation and test
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)


# Predict and print reports
y_val_pred = nb_classifier.predict(x_val)
y_val_proba = nb_classifier.predict_proba(x_val)[:, 1]

print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))
print("Validation ROC/AUC Score:", roc_auc_score(y_val, y_val_proba))
print("\033[34m--------------------------------------------------------------------------------\033[0m")

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'Naive_Bayes (AUC = {roc_auc_score(y_val, y_val_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')

# Evaluate the model on the test set
y_test_pred = nb_classifier.predict(x_test)
y_test_proba = nb_classifier.predict_proba(x_test)[:, 1]

print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))
print("Test ROC/AUC Score:", roc_auc_score(y_test, y_test_proba))
print("\033[34m--------------------------------------------------------------------------------\033[0m")

# Plot the confusion matrix
ConfusionMatrixDisplay.from_estimator(nb_classifier, x_test, y_test)
plt.show()
