from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd

X_train = pd.read_csv("x_train_final.csv")
X_test = pd.read_csv("x_test_final.csv")

train_list = X_train['train'].to_list() + X_test['train'].to_list()
gare_list = X_train['gare'].to_list() + X_test["gare"].to_list()
date_list = X_train["date"].to_list() + X_test["date"].to_list()

# Initialize the LabelEncoder
TrainEncoder = LabelEncoder()
# Fit and transform the 'trains' column
TrainEncoder.fit(train_list)

# Initialize the LabelEncoder
GareEncoder = LabelEncoder()
# Fit and transform the 'gare' column
GareEncoder.fit(gare_list)

# Initialize the LabelEncoder
DateEncoder = LabelEncoder()
# Fit and transform the 'gare' column
DateEncoder.fit(date_list)