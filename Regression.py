import pandas as pd
from format_x import preprocess_df
from sklearn.linear_model import LinearRegression

def main():
    # --- Training Phase ---
    # Read the training data
    X_train = pd.read_csv("x_train_pca.csv")

    # Read the target variable (assuming the target is in the first column)
    y_train = pd.read_csv("y_train_final_j5KGWWK.csv").iloc[:, 0]

    print("Training features shape:", X_train.shape)
    print("Training target shape:", y_train.shape)

    # Fit a regression model (here using a simple LinearRegression)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Prediction Phase ---
    # Read and preprocess the test data
    df_test = pd.read_csv("x_test_final.csv")
    X_test = preprocess_df(df_test)

    # Ensure the test DataFrame has the same features as the training set.
    # This aligns one-hot encoded columns (new categories in test are set to 0).
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Predict using the fitted model
    y_pred = model.predict(X_test)

    # --- Save Predictions ---
    # Read the sample submission file to mimic the required format.
    # This file typically contains an ID column that should be preserved.
    submission = pd.read_csv("y_sample_final.csv")
    # If the sample file does not have an ID column, we create one from the index.
    if 'id' not in submission.columns:
        submission['id'] = submission.index

    # Assign predictions to the submission (assuming the target column is named 'y')
    submission['y'] = y_pred

    # Save the submission file (adjust the file name as needed)
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")