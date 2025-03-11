from sklearn.preprocessing import LabelEncoder
import pandas as pd

X_train = pd.read_csv("x_train_final.csv")
X_test = pd.read_csv("x_test_final.csv")

train_list = X_train['train'].to_list() + X_test['train'].to_list()
# Initialize the LabelEncoder
LE = LabelEncoder()

# Fit and transform the 'trains' column
LE.fit(train_list)