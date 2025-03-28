import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from encoders import TrainEncoder, GareEncoder, DateEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

X_initial = pd.read_csv("x_train_no_outlier.csv")
y = pd.read_csv("y_train_no_outlier.csv").iloc[:, 2]

X_initial["gare"] = GareEncoder.transform(X_initial['gare'])
X_initial["train"] = TrainEncoder.transform(X_initial['train'])
X_initial["new_date"] = DateEncoder.transform(X_initial["date"])

X_initial['date'] = pd.to_datetime(X_initial['date'], errors='coerce')
# Extract additional date features
X_initial['year'] = X_initial['date'].dt.year
X_initial['month'] = X_initial['date'].dt.month
X_initial['day'] = X_initial['date'].dt.day
X_initial['dayofweek'] = X_initial['date'].dt.dayofweek
X_initial['weekofyear'] = X_initial['date'].dt.isocalendar().week.astype(int)


# Define Features
features = ["new_date" ,"gare", "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"] #'year', 'weekofyear', 'dayofweek', "train",

X = X_initial[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=203)

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_val, y_val, enable_categorical=True)

# Define hyperparameters
params = {
    "objective": "reg:absoluteerror",
    "device": "cuda",
    "booster": "gbtree",
    "subsample":0.8,
    "learning_rate":0.12,
    "eval_metric":"mae",
}

n = 2500
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
)

# Predictions
y_train_pred = model.predict(dtrain_reg)
y_test_pred = model.predict(dtest_reg)

# Compute the Mean Absolute Error on the training set
mae_train = mean_absolute_error(y_train, y_train_pred)
print("Training Set MAE:", mae_train)

mae_test = mean_absolute_error(y_val, y_test_pred)
print("Test Set MAE:", mae_test)



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
dtest_final = xgb.DMatrix(X_test_scaled, enable_categorical=True)

# Predict using the fitted model on scaled test data
y_pred = model.predict(dtest_final)
df_out = pd.DataFrame(y_pred, columns=["p0q0"])
df_out.to_csv("xgboost_no_new.csv")