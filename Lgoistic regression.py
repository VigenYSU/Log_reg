import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=100000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.c_[np.ones((n_samples, 1)), X]
        self.theta = np.zeros((X_b.shape[1], 1))

        for i in range(self.n_iterations):
            linear_output = X_b.dot(self.theta)
            y_pred = self.sigmoid(linear_output)
            error = y_pred - y
            gradients = X_b.T.dot(error) / n_samples
            self.theta -= self.learning_rate * gradients

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_b.dot(self.theta))

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# Load and prepare data
data = r"C:\\Users\\Vigen\\Downloads\\weatherAUS.csv\\weatherAUS.csv"
df = pd.read_csv(data)
df.columns = df.columns.str.strip()

# Use selected features and target
df = df[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'RainToday', 'RainTomorrow']]
df = df.dropna()
df['RainToday'] = (df['RainToday'] == 'Yes').astype(int)
df['RainTomorrow'] = (df['RainTomorrow'] == 'Yes').astype(int)

# Features and labels
X = df[['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm']].values
y = df['RainTomorrow'].values.reshape(-1, 1)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train model
model = LogisticRegression(learning_rate=0.1, n_iterations=10000)
model.fit(X_train, y_train)

# Predict
threshold = 0.5
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test, threshold=threshold)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Rain", "Rain"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Threshold = {:.2f})".format(threshold))
plt.grid(False)
plt.show()

# Classification report
print("\nClassification Report (Threshold = {:.2f}):\n".format(threshold))
print(classification_report(y_test, predictions, target_names=["No Rain", "Rain"]))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve for Rain Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

# Print thresholds vs TPR and FPR
print("\nThreshold vs TPR (Recall) and FPR:")
for i in range(0, len(thresholds), len(thresholds) // 5):
    print(f"Threshold = {thresholds[i]:.2f}, TPR = {tpr[i]:.2f}, FPR = {fpr[i]:.2f}")
