import pandas as pd
import numpy as np
from encoders import TrainEncoder, GareEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Data Loading and Preparation
# ---------------------------

# Load training features and target
X_initial = pd.read_csv("x_train_final.csv")
y = pd.read_csv("y_train_final_j5KGWWK.csv").iloc[:, 1]  # adjust if the target is not the first column

X_initial["gare"] = GareEncoder.transform(X_initial['gare'])
X_initial["train"] = TrainEncoder.transform(X_initial['train'])

X_initial['date'] = pd.to_datetime(X_initial['date'], errors='coerce')
# Extract additional date features
X_initial['year'] = X_initial['date'].dt.year
X_initial['month'] = X_initial['date'].dt.month
X_initial['day'] = X_initial['date'].dt.day
X_initial['dayofweek'] = X_initial['date'].dt.dayofweek
X_initial['weekofyear'] = X_initial['date'].dt.isocalendar().week.astype(int)


# Choose features wisely:
# Here, we select some numeric process features and include transformed time features if available.
# You can adjust the feature list based on your domain knowledge.
selected_features = ['year', 'month', 'day', "gare", "train", 'p2q0', 'p3q0', 'p4q0']#'year', 'month', 'day', 'dayofweek', 'weekofyear',  'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4'

# Create feature matrix X and target vector y
X = X_initial[selected_features].values
y = y.values

# Normalize features to help the LSTM converge
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------------------
# Sequence Creation
# ---------------------------
# For time series forecasting with LSTM, we need to create sequences.
# Here, each sequence is composed of 'lookback' consecutive timesteps.
lookback = 9  # we take the half mean arret per gare

def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, lookback)

# Split the sequences into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print("Training sequences shape:", X_train.shape)
print("Validation sequences shape:", X_val.shape)

# ---------------------------
# LSTM Model Building
# ---------------------------
# We build a simple LSTM network with one LSTM layer, dropout for regularization,
# and a dense output layer to predict a continuous target.
model = Sequential()
model.add(LSTM(50, activation='elu', input_shape=(lookback, X_seq.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ---------------------------
# Model Training
# ---------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1  # This enables verbose logging for tracking training progress
)

# ---------------------------
# Evaluation on Training Set
# ---------------------------
y_train_pred = model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
print("Training MAE:", mae_train)




# --- Prediction Phase ---

def create_test_sequences(X_test, lookback, initial_data=None):
    """
    Create sequences for a test set.

    Parameters:
      X_test (np.array): Test set features.
      lookback (int): Number of timesteps to include in each sequence.
      initial_data (np.array): Optional array of shape (lookback, n_features) to prepend to X_test.

    Returns:
      np.array: Array of test sequences with shape (n_sequences, lookback, n_features).
    """
    # If provided, prepend the initial_data (e.g., tail of training set) to the test data.
    if initial_data is not None:
        X_full = np.concatenate([initial_data, X_test], axis=0)
    else:
        X_full = X_test

    sequences = []
    # Note: We use len(X_full) - lookback + 1 so that the first sequence starts at index 0
    for i in range(len(X_full) - lookback + 1):
        sequences.append(X_full[i:i + lookback])
    return np.array(sequences)


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


df_test = X_test[['year', 'month', 'day', "gare", "train", 'p2q0', 'p3q0', 'p4q0']] #'year', 'month', 'day', 'dayofweek', 'weekofyear', , 'p0q2', 'p0q3', 'p0q4'

# Scale the test features using the same scaler fitted on the training data
X_test_scaled = scaler.transform(df_test)

X_test_sequences = create_test_sequences(df_test, lookback, initial_data=X[-1*(lookback-1):])

# Predict using the fitted model on scaled test data
y_pred = model.predict(X_test_sequences)
df_out = pd.DataFrame(y_pred, columns=["p0q0"])
df_out.to_csv("lstmall2.csv")