# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = '/content/Marketing_data.csv'
data = pd.read_csv(file_path)

# Step 2: Preprocess the data (one-hot encoding for categorical variables)
data_encoded = pd.get_dummies(data, drop_first=True)

# Step 3: Separate features (X) and target (y)
X = data_encoded.drop('deposit_yes', axis=1)  # Target variable (1 for yes, 0 for no)
y = data_encoded['deposit_yes']

# Step 4: Split the data into training and testing sets
def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split_idx = int(len(y) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the features
def standardize_data(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std

X_train_scaled, mean, std = standardize_data(X_train)
X_test_scaled = (X_test - mean) / std

# Step 6: Implement a simple linear SVM model
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
svm.fit(X_train_scaled.to_numpy(), y_train.to_numpy())

# Step 7: Make predictions on the test set
y_pred = svm.predict(X_test_scaled.to_numpy())

# Step 8: Evaluate the model performance
accuracy = np.mean(y_pred == np.where(y_test <= 0, -1, 1))
conf_matrix = np.zeros((2, 2))
for true, pred in zip(y_test, y_pred):
    conf_matrix[int(true), int(pred > 0)] += 1

# Print evaluation metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
plt.matshow(conf_matrix, cmap='Blues', alpha=0.7)
for (i, j), val in np.ndenumerate(conf_matrix):
    plt.text(j, i, f"{int(val)}", ha='center', va='center')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# Step 9: Plot decision boundary (if dataset has 2 features for visualization)
if X_train_scaled.shape[1] == 2:
    X_combined = np.vstack((X_train_scaled, X_test_scaled))
    y_combined = np.hstack((y_train, y_test))

    x_min, x_max = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
    y_min, y_max = X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X_combined[:, 0], X_combined[:, 1], c=y_combined, edgecolors='k', cmap='coolwarm')
    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1 (Standardized)")
    plt.ylabel("Feature 2 (Standardized)")
    plt.show()
