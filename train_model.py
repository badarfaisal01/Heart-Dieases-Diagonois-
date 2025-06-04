import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier

# Load the dataset
df = pd.read_csv("data/heart.csv")

# Separate features and target
X = df.drop("target", axis=1).values
y = df["target"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to address class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

##########
# Manual Logistic Regression Model with Gradient Descent
##########

np.random.seed(42)
n_features = X.shape[1]
weights = np.random.randn(n_features)
bias = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training parameters
learning_rate = 0.01
epochs = 1000

print("Training manual logistic regression model with gradient descent...")
for epoch in range(epochs):
    linear_output = np.dot(X_train_resampled, weights) + bias
    predictions = sigmoid(linear_output)

    error = predictions - y_train_resampled
    dw = np.dot(X_train_resampled.T, error) / len(X_train_resampled)
    db = np.mean(error)

    weights -= learning_rate * dw
    bias -= learning_rate * db

    if epoch % 100 == 0:
        pred_labels = (predictions > 0.5).astype(int)
        acc = accuracy_score(y_train_resampled, pred_labels)
        print(f"Epoch {epoch}, Training Accuracy: {acc:.4f}")

# Evaluate manual model on test data
test_preds = sigmoid(np.dot(X_test, weights) + bias)
test_labels = (test_preds > 0.5).astype(int)
manual_test_accuracy = accuracy_score(y_test, test_labels)
print(f"Manual Logistic Regression Test Accuracy: {manual_test_accuracy:.4f}")

##########
# Scikit-learn Models Training and Evaluation
##########

# Define models to train
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42),
    "Neural Network": MLPClassifier(random_state=42, max_iter=2000, learning_rate_init=0.001, hidden_layer_sizes=(50, 50))
}
best_model = None
best_accuracy = manual_test_accuracy  # Initialize with manual model accuracy
best_model_name = "Manual Logistic Regression"

print("\nTraining other models with scikit-learn...")

for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Test Accuracy: {acc:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Hyperparameter tuning using Grid Search for the best model
param_grid = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200]},
    'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
    'Support Vector Machine': {'C': [0.1, 1, 10]},
    'Neural Network': {'hidden_layer_sizes': [(10,), (50,), (100,)]}
}

for name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[name], cv=5)
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} with Grid Search Test Accuracy: {acc:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name

# Prepare model to save: if best is the manual model, save weights + scaler dict
if best_model_name == "Manual Logistic Regression":
    model_to_save = {
        "weights": weights,
        "bias": bias,
        "scaler": scaler
    }
else:
    # Save the best sklearn model + scaler inside a dictionary
    model_to_save = {
        "model": best_model,
        "scaler": scaler
    }

# Save the model
joblib.dump(model_to_save, "models/best_heart_model.pkl")
print(f"\nBest model: {best_model_name} with Test Accuracy: {best_accuracy:.4f}")
print("Model saved to models/best_heart_model.pkl")

# Evaluate the best model on the test set
X_test_scaled = scaler.transform(X_test)
y_pred = best_model.predict(X_test_scaled) if best_model_name != "Manual Logistic Regression" else (sigmoid(np.dot(X_test_scaled, weights) + bias) > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))