import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from encoders import TrainEncoder, GareEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt

# Load training features and target
X_initial = pd.read_csv("x_train_no_outlier.csv")
y = pd.read_csv("y_train_no_outlier.csv").iloc[:, 2]

# Transform categorical features using custom encoders
X_initial["gare"] = GareEncoder.transform(X_initial['gare'])
X_initial["train"] = TrainEncoder.transform(X_initial['train'])

# Convert date to datetime and extract additional date features
X_initial['date'] = pd.to_datetime(X_initial['date'], errors='coerce')
X_initial['year'] = X_initial['date'].dt.year
X_initial['month'] = X_initial['date'].dt.month
X_initial['day'] = X_initial['date'].dt.day
X_initial['dayofweek'] = X_initial['date'].dt.dayofweek
X_initial['weekofyear'] = X_initial['date'].dt.isocalendar().week.astype(int)

# Select features to be used in the model
X = X_initial[['year', 'weekofyear', 'dayofweek', "train", "gare", "arret",
               "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the XGBoost regressor with fixed n_estimators and subsample, tuning only learning_rate.
xgb_reg = xgb.XGBRegressor(objective="reg:absoluteerror", random_state=42)

# Define a grid of hyperparameters for tuning (only learning_rate is varied)
param_grid = {
    "n_estimators": [1000],  # Fixed value
    "learning_rate": [0.05, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.2,],
    "subsample": [0.6],  # Fixed value
    "tree_method": ["gpu_hist"],
    "booster": ["gbtree"],
}

# Setup GridSearchCV with 5-fold cross-validation.
# Using scoring="neg_mean_absolute_error" so we convert negatives to positive MAE later.
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5,
                           scoring="neg_mean_absolute_error", verbose=1, n_jobs=-1,
                           return_train_score=True)

# Run the grid search on the training data
grid_search.fit(X_train, y_train)

# Display the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Extract the learning_rate values and corresponding train/test MAE scores
learning_rates = [params['learning_rate'] for params in grid_search.cv_results_['params']]
# Since the scoring returns negative MAE, we take the negative to get positive error values
mean_train_mae = -grid_search.cv_results_['mean_train_score']
mean_test_mae = -grid_search.cv_results_['mean_test_score']

# Plot the MAE results vs. learning_rate
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, mean_train_mae, marker='o', label='Train MAE')
plt.plot(learning_rates, mean_test_mae, marker='s', label='Test MAE')
plt.xlabel('Learning Rate')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Grid Search Results: MAE vs. Learning Rate')
plt.legend()
plt.grid(True)
plt.savefig("grid_search_mae_vs_learning_rate.png")
# plt.show()
