import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Read the dataset
df = pd.read_csv('Dataset_1000.csv')
print("--------------------Exploratory Data analysis----------------------")
print(df.head())
print("Information about dataset:")
print(df.info())

# Select subset of columns for analysis
subset_columns = ['player_height', 'player_weight', 'pts', 'ts_pct', 'oreb_pct', 'dreb_pct', 'gp']

# Data analysis
sns.pairplot(df[subset_columns])

print(" ")
plt.suptitle('Pair Plot of Selected Variables', y=1.02)
plt.show()

# Define features and target variable
X = df[['player_height', 'player_weight', 'oreb_pct', 'gp', 'ts_pct','reb','usg_pct','ast_pct']]
y = df['pts']

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)


class BayesianLinearRegression:
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_cov = None

    def fit(self, X, y):
        X = tf.concat([tf.ones((tf.shape(X)[0], 1), dtype=tf.float64), X], axis=1)  # Add bias term
        D = X.shape[1]
        XtX = tf.matmul(tf.transpose(X), X)
        self.w_cov = tf.linalg.inv(self.alpha * tf.eye(D, dtype=tf.float64) + self.beta * XtX)
        self.w_mean = self.beta * tf.matmul(tf.matmul(self.w_cov, tf.transpose(X)), tf.expand_dims(y, axis=1))

    def predict(self, X):
        X = tf.concat([tf.ones((tf.shape(X)[0], 1), dtype=tf.float64), X], axis=1)  # Add bias term
        prediction = tf.matmul(X, self.w_mean)
        return tf.squeeze(prediction, axis=1).numpy()

# Instantiate the BayesianLinearRegression model
model = BayesianLinearRegression(alpha=1, beta=1)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the target variable for the test data
print("Result of Model prediction: ")
y_pred = model.predict(X_test)
print(y_pred)

print("-----------------------VISUALIZATION OF RESULT-------------------------")
print(" ")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
plt.title('Actual vs Predicted')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_pred, bins=30, kde=True, color='red')
plt.title('Distribution of Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


print("-----------------------------MODEL EVALUATION-------------------------------")
print(" ")

# Calculate Mean Squared Error
mse = tf.reduce_mean(tf.square(y_test - y_pred)).numpy()
print("Mean Squared Error:", mse)
# Calculate Mean Absolute Error
mae = tf.reduce_mean(tf.abs(y_test - y_pred)).numpy()
print("Mean Absolute Error:", mae)

# Calculate Root Mean Squared Error
rmse = tf.sqrt(mse).numpy()
print("Root Mean Squared Error:", rmse)
# Calculate R-squared score
def r_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean)**2)
    rss = np.sum((y_true - y_pred)**2)
    r2 = 1 - (rss / tss)
    return r2

r_squared = r_score(y_test, y_pred)
print("R-squared score:", r_squared)
