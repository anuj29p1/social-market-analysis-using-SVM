import numpy as np

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

        # Converting labels to +1, -1
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

# Example usage:
# Create synthetic data for testing
np.random.seed(1)
X = np.random.randn(50, 2)  # 50 samples, 2 features
y = np.array([1 if x[0] + x[1] > 0 else 0 for x in X])  # Generate binary labels

# Instantiate and train the SVM
svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)

# Make predictions
predictions = svm.predict(X)

# Evaluate
accuracy = np.mean(predictions == np.where(y <= 0, -1, 1))
print(f"Accuracy: {accuracy * 100:.2f}%")
