# import important Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# get the Data 
df=pd.read_csv('all_seasons.csv')

df

df.head()

df.info()

# Exploratory data anlysis 
subset_columns = ['age', 'player_height', 'player_weight', 'pts', 'reb', 'ast']

# Create a pair plot
sns.pairplot(df[subset_columns])
plt.suptitle('Pair Plot of Selected Variables', y=1.02)
plt.show()


plt.figure(figsize=(12, 8))
sns.histplot(df['pts'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Points')
plt.xlabel('Points')
plt.ylabel('Frequency')
plt.show()


# Training and Testing DataSets

X=df[['player_height','player_weight','gp']]
y=df[['pts']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)


# train the model 

class BayesianLinearRegression:
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha  # Prior precision for weights
        self.beta = beta    # Prior precision for noise
        self.w_mean = None  # Mean of weight posterior
        self.w_cov = None   # Covariance of weight posterior

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        D = X.shape[1]
        self.w_cov = np.linalg.inv(self.alpha * np.eye(D) + self.beta * X.T @ X)
        self.w_mean = self.beta * self.w_cov @ X.T @ y

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        return X @ self.w_mean

model=BayesianLinearRegression()

model.fit(X_train,y_train)

# Predicting test data

y_pred=model.predict(X_test)
y_pred

# Evaluation of Model 

def mean_square_error(y_true,y_pred):
    error=(y_true-y_pred) ** 2
    mse=np.mean(error)
    return mse

mse=mean_square_error(y_test,y_pred)

mse

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Line')
plt.title('Actual vs Predicted')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.grid(True)
plt.show()








