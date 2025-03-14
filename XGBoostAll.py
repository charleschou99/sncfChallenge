import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from encoders import TrainEncoder, GareEncoder
from sklearn.preprocessing import StandardScaler

# Load training features and target
X_initial = pd.read_csv("x_train_final.csv")
y = pd.read_csv("y_train_final_j5KGWWK.csv").iloc[:, 1]

X_initial["gare"] = GareEncoder.transform(X_initial['gare'])
X_initial["train"] = TrainEncoder.transform(X_initial['train'])

X_initial['date'] = pd.to_datetime(X_initial['date'], errors='coerce')
# Extract additional date features
X_initial['year'] = X_initial['date'].dt.year
X_initial['month'] = X_initial['date'].dt.month
X_initial['day'] = X_initial['date'].dt.day
X_initial['dayofweek'] = X_initial['date'].dt.dayofweek
X_initial['weekofyear'] = X_initial['date'].dt.isocalendar().week.astype(int)

X = X_initial[['year', 'weekofyear', 'dayofweek', "train", "gare", "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize XGBoost regressor
xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Define a grid of hyperparameters for tuning
param_grid = {
    "n_estimators": [100, 500],
    "max_depth": [15],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0]
}

# Setup GridSearchCV with 3-fold cross-validation; verbose=1 enables progress output
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5,
                           scoring="neg_mean_squared_error", verbose=1, n_jobs=-1)

# Run the grid search on the training data
grid_search.fit(X_train, y_train)

# Display the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Predict on the training set using the best model
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_val)

# Compute the Mean Squared Error (MSE) on the training set
mse_train = mean_squared_error(y_train, y_train_pred)
print("Training Set MSE:", mse_train)

mse_test = mean_squared_error(y_val, y_test_pred)
print("Test Set MSE:", mse_test)

# --- Prediction Phase ---
# Read and preprocess the test data
X_test = pd.read_csv("x_test_final.csv")
# X_test["train"] = TrainEncoder.transform(X_test['train'])
X_test["gare"] = GareEncoder.transform(X_test['gare'])
X_test["train"] = TrainEncoder.transform(X_test['train'])

X_test['date'] = pd.to_datetime(X_test['date'], errors='coerce')
# Extract additional date features
X_test['year'] = X_test['date'].dt.year
X_test['month'] = X_test['date'].dt.month
X_test['day'] = X_test['date'].dt.day
X_test['dayofweek'] = X_test['date'].dt.dayofweek
X_test['weekofyear'] = X_test['date'].dt.isocalendar().week.astype(int)

df_test = X_test[['year', 'weekofyear', 'dayofweek', "train", "gare", "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]]
# Reindex to ensure the test set has the same columns as the training set
df_test = df_test.reindex(columns=df_test.columns, fill_value=0)

# Scale the test features using the same scaler fitted on the training data
X_test_scaled = scaler.transform(df_test)

# Predict using the fitted model on scaled test data
y_pred = best_model.predict(X_test_scaled)
df_out = pd.DataFrame(y_pred, columns=["p0q0"])
df_out.to_csv("xgboost_all_submission.csv")