import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from encoders import TrainEncoder, GareEncoder, DateEncoder
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------

# Load training features and target
X_initial = pd.read_csv("x_train_final.csv")
y = pd.read_csv("y_train_final_j5KGWWK.csv").iloc[:, 1]

# Transform categorical features using custom encoders
X_initial["gare"] = GareEncoder.transform(X_initial['gare'])
X_initial["train"] = TrainEncoder.transform(X_initial['train'])
X_initial["new_date"] = DateEncoder.transform(X_initial["date"])

# Convert 'date' to datetime and extract additional date features
X_initial['date'] = pd.to_datetime(X_initial['date'], errors='coerce')
X_initial['year'] = X_initial['date'].dt.year
X_initial['month'] = X_initial['date'].dt.month
X_initial['day'] = X_initial['date'].dt.day
X_initial['dayofweek'] = X_initial['date'].dt.dayofweek
X_initial['weekofyear'] = X_initial['date'].dt.isocalendar().week.astype(int)

# Select features to be used in the model
features = ["gare", "arret", "new_date",
            "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]

X = X_initial[features]

# Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------
# CatBoost Model Training
# ---------------------------

# Initialize CatBoostRegressor with chosen hyperparameters.
# Using loss_function='MAE' to match our evaluation metric.
model = CatBoostRegressor(
    loss_function='MAE',
    iterations=3500,
    learning_rate=0.1,
    verbose=500,
    early_stopping_rounds=100,
    eval_metric="MAE",
    # bootstrap_type="MVS",
    subsample=0.9,
    max_depth=6
)

# Train the model, using the validation set for early stopping
model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

# ---------------------------
# Evaluation
# ---------------------------

# Predict on training and validation sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Compute the Mean Absolute Error (MAE) on the training and validation sets
train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)

print("Training MAE:", train_mae)
print("Validation MAE:", val_mae)


# --- Prediction Phase ---
# Read and preprocess the test data
X_test = pd.read_csv("x_test_final.csv")
# X_test["train"] = TrainEncoder.transform(X_test['train'])
X_test["gare"] = GareEncoder.transform(X_test['gare'])
X_test["train"] = TrainEncoder.transform(X_test['train'])
X_test["new_date"] = DateEncoder.transform(X_test["date"])

X_test['date'] = pd.to_datetime(X_test['date'], errors='coerce')
# Extract additional date features
X_test['year'] = X_test['date'].dt.year
X_test['month'] = X_test['date'].dt.month
X_test['day'] = X_test['date'].dt.day
X_test['dayofweek'] = X_test['date'].dt.dayofweek
X_test['weekofyear'] = X_test['date'].dt.isocalendar().week.astype(int)


df_test = X_test[features]
# Reindex to ensure the test set has the same columns as the training set
df_test = df_test.reindex(columns=df_test.columns, fill_value=0)

# Scale the test features using the same scaler fitted on the training data
X_test_scaled = scaler.transform(df_test)

# Predict using the fitted model on scaled test data
y_pred = model.predict(X_test_scaled)
df_out = pd.DataFrame(y_pred, columns=["p0q0"])
df_out.to_csv("catboost_new.csv")