# Predictive Modeling Project

This project implements several predictive modeling techniques—including Ridge Regression, LSTM, XGBoost, CatBoost approach for forecasting a target variable. The repository includes scripts for data preprocessing, model training, hyperparameter tuning, and predictions.

## Project Structure

- **Models/**: Contains the Python scripts for each model:
  - Ridge Regression
  - LSTM
  - XGBoost
  - Catboost
- **Data/**: Folder with dataset CSV files.
- **Notebooks/**: Jupyter Notebooks for data visualization and outlier detection (e.g., `visualisation.ipynb` and `outlier_detection.ipynb`).---
- **Submission/**: Submission csv.

## How to Use the Scripts

1. **Prepare Data:**
   - Ensure you have the required CSV files (e.g., `x_train_no_outlier.csv`, `y_train_no_outlier.csv`, `x_test_final.csv`, etc.).
   - Place these files in your local directory (or in the `data/` folder if you prefer).

2. **Update File Paths:**
   - Open each script in the `models/` folder and locate the lines where data is loaded using `pd.read_csv()`.
   - Modify the file paths to point to the correct location of your CSV files.

3. **Run the Models:**
   - Navigate to the `models/` folder in your terminal or command prompt.
   - Run the desired model script. For example, to run the Ridge Regression model:
     ```
     python RegressionAll.py
     ```

4. **Visualizations and Outlier Detection:**
   - Open the Jupyter Notebooks in the `notebooks/` folder (`visualisation.ipynb` and `outlier_detection.ipynb`) to explore the data, visualize distributions, and examine outlier detection results.
   - Run the cells sequentially in Jupyter Notebook or JupyterLab.

## Requirements

- Python 3.6 or higher
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `catboost`
  - `tensorflow` (for LSTM)
  - `matplotlib`
  - `seaborn`

## Customization

- **CSV File Paths:**  
  Modify the CSV file paths in each script as necessary to match your local file structure.

- **Model Hyperparameters:**  
  Each script contains parameters that you can adjust to suit your dataset and experimentation needs.