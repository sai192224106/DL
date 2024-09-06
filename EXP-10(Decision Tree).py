import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Decision Tree model
classifier = DecisionTreeClassifier(criterion='entropy', random_state=5)
classifier.fit(X_train, y_train)

# Display the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(classifier, filled=True, rounded=True, feature_names=data.feature_names)
plt.show()

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Display the results (confusion matrix and accuracy)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
