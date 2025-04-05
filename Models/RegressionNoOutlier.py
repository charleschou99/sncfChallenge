import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from encoders import TrainEncoder, GareEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


# --- Training Phase ---
# Read the training data
X_train = pd.read_csv("x_train_no_outlier.csv")
# X_train["train"] = TrainEncoder.transform(X_train['train'])
X_train["gare"] = GareEncoder.transform(X_train['gare'])

X_train['date'] = pd.to_datetime(X_train['date'], errors='coerce')
# Extract additional date features
X_train['year'] = X_train['date'].dt.year
X_train['month'] = X_train['date'].dt.month
X_train['day'] = X_train['date'].dt.day
X_train['dayofweek'] = X_train['date'].dt.dayofweek
X_train['weekofyear'] = X_train['date'].dt.isocalendar().week.astype(int)

df_train = X_train[['weekofyear', 'dayofweek', "gare", "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]]
y_train = pd.read_csv("y_train_no_outlier.csv").iloc[:, 2]

print("Training features shape:", df_train.shape)
print("Training target shape:", y_train.shape)

# Normalize the training features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train)

# --- Define the Best Linear Model ---
# Here we use RidgeCV, which performs Ridge regression and selects the best regularization parameter (alpha)
alphas = [0.1, 0.5, 1.0, 5.0, 10.0]  # candidate regularization strengths
model = RidgeCV(alphas=alphas, cv=5)

# Perform cross validation on the scaled training data
cv_scores = cross_val_score(model, X_train_scaled, y_train, scoring="neg_mean_squared_error", cv=5)
mse_cv = -np.mean(cv_scores)
rmse_cv = np.sqrt(mse_cv)
print("Cross Validation MSE:", mse_cv)
print("Cross Validation RMSE:", rmse_cv)

# Fit the model on the full scaled training data
model.fit(X_train_scaled, y_train)

# --- Prediction Phase ---
# Read and preprocess the test data
X_test = pd.read_csv("x_test_final.csv")
# X_test["train"] = TrainEncoder.transform(X_test['train'])
X_test["gare"] = GareEncoder.transform(X_test['gare'])
df_test = X_test[["gare", "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]]
# Reindex to ensure the test set has the same columns as the training set
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

# Scale the test features using the same scaler fitted on the training data
X_test_scaled = scaler.transform(df_test)

# Predict using the fitted model on scaled test data
y_pred = model.predict(X_test_scaled)
df_out = pd.DataFrame(y_pred, columns=["p0q0"])
df_out.to_csv("ridge_first_submission.csv")

# Optionally, if you have the true test target values, you could compute test metrics as well:
y_test = pd.read_csv("y_test_final.csv").iloc[:, 2]  # adjust column index as needed
mae_test = mean_absolute_error(y_test, y_pred)
print("Test MAE:", mae_test)
