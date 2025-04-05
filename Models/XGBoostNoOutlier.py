# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
from functools import partial

# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Classifier/Regressor
from xgboost import XGBRegressor, DMatrix

# Model selection
from sklearn.model_selection import KFold, StratifiedKFold

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer

# Data processing
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from encoders import GareEncoder, TrainEncoder, DateEncoder

# Load training features and target
X_initial = pd.read_csv("x_train_winsorize_new.csv")
y = pd.read_csv("y_train_winsorize_new.csv").iloc[:, 2]

# Transform categorical features using custom encoders
X_initial["gare"] = GareEncoder.transform(X_initial['gare'])
X_initial["train"] = TrainEncoder.transform(X_initial['train'])
X_initial["date_now"] = DateEncoder.transform(X_initial['date'])

# Frquency encoding
# Calculate the frequency (count) for each category in "gare"
gare_freq = X_initial['gare'].value_counts()
X_initial['gare_freq'] = X_initial['gare'].map(gare_freq)

# Convert date to datetime and extract additional date features
# X_initial['date'] = pd.to_datetime(X_initial['date'], errors='coerce')
# X_initial['year'] = X_initial['date'].dt.year
# X_initial['month'] = X_initial['date'].dt.month
# X_initial['day'] = X_initial['date'].dt.day
# X_initial['dayofweek'] = X_initial['date'].dt.dayofweek
# X_initial['weekofyear'] = X_initial['date'].dt.isocalendar().week.astype(int)

# Select features to be used in the model

features = ["date_now", "gare_freq", "arret",
               "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]

X = X_initial[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reporting util for different optimizers
def report_perf(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers

    optimizer = a sklearn or a skopt optimizer
    X = the training set
    y = our target
    title = a string label for the experiment
    """
    start = time()

    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1" + " %.3f") % (time() - start,
                                     len(optimizer.cv_results_['params']),
                                     best_score,
                                     best_score_std))
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params

# Setting the scoring function
scoring = make_scorer(partial(mean_absolute_error),
                      greater_is_better=False)

# Setting the validation strategy
skf = StratifiedKFold(n_splits=7,
                      shuffle=True,
                      random_state=0)

cv_strategy = list(skf.split(X, y))

# Setting the basic regressor
reg = XGBRegressor(random_state=0, booster='gbtree', objective='reg:absoluteerror', device='cuda')

# Setting the search space
search_spaces = {'learning_rate': Real(0.01, 0.2, 'uniform'),
                 'max_depth': Integer(2, 12),
                 'subsample': Real(0.1, 1.0, 'uniform'),
                 'colsample_bytree': Real(0.1, 1.0, 'uniform'), # subsample ratio of columns by tree
                 # 'reg_lambda': Real(1e-9, 100., 'uniform'), # L2 regularization
                 # 'reg_alpha': Real(1e-9, 100., 'uniform'), # L1 regularization
                 'n_estimators': Integer(50, 5000)
   }

# Wrapping everything up into the Bayesian optimizer
opt = BayesSearchCV(estimator=reg,
                    search_spaces=search_spaces,
                    scoring=scoring,
                    cv=cv_strategy,
                    n_iter=120,                                       # max number of trials
                    n_points=1,                                       # number of hyperparameter sets evaluated at the same time
                    n_jobs=-1,                                         # number of jobs
                    iid=False,                                        # if not iid it optimizes on the cv score
                    return_train_score=False,
                    refit=False,
                    optimizer_kwargs={'base_estimator': 'GP'},        # optmizer parameters: we use Gaussian Process (GP)
                    random_state=203)                                   # random state for replicability


# Running the optimizer
overdone_control = DeltaYStopper(delta=0.0001)                    # We stop if the gain of the optimization becomes too small
time_limit_control = DeadlineStopper(total_time=60*60*4)          # We impose a time limit (7 hours)

best_params = report_perf(opt, X, y,'XGBoost_regression',
                          callbacks=[overdone_control, time_limit_control])

# Best model
# XGBoost_regression took 13806.69 seconds,  candidates checked: 59, best CV score: -0.607 ± 0.001
# Best parameters:
# OrderedDict([('colsample_bytree', 0.5907540767609287),
#              ('learning_rate', 0.10418554981975407),
#              ('max_depth', 9),
#              ('n_estimators', 5000),
#              ('reg_alpha', 57.047542948881876),
#              ('reg_lambda', 100.0),
#              ('subsample', 0.8507596698572524)])

# Prediction
# Transferring the best parameters to our basic regressor
reg = XGBRegressor(random_state=0, booster='gbtree', objective='reg:absoluteerror', tree_method='gpu_hist', **best_params)

# Read and preprocess the test data
X_test = pd.read_csv("x_test_final.csv")
# X_test["train"] = TrainEncoder.transform(X_test['train'])
X_test["gare"] = GareEncoder.transform(X_test['gare'])
X_test["train"] = TrainEncoder.transform(X_test['train'])
X_test["date_now"] = DateEncoder.transform(X_test["date"])

gare_freq = X_test['gare'].value_counts()
X_test['gare_freq'] = X_test['gare'].map(gare_freq)

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

# Cross-validation prediction
folds = 10
skf = StratifiedKFold(n_splits=folds,
                      shuffle=True,
                      random_state=0)

predictions = np.zeros(len(df_test))
mae = list()

for k, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    reg.fit(X.iloc[train_idx, :], y[train_idx])
    val_preds = reg.predict(X.iloc[val_idx, :])
    val_mae = mean_absolute_error(y_true=y[val_idx], y_pred=val_preds)
    print(f"Fold {k} MAE: {val_mae:0.5f}")
    mae.append(val_mae)
    predictions += reg.predict(df_test).ravel()

predictions /= folds
print(f"repeated CV MAE: {np.mean(mae):0.5f} (std={np.std(mae):0.5f})")

# Fold 0 RMSE: 0.66366
# Fold 1 RMSE: 0.66130
# Fold 2 RMSE: 0.66371
# Fold 3 RMSE: 0.66346
# Fold 4 RMSE: 0.66141
# Fold 5 RMSE: 0.66228
# Fold 6 RMSE: 0.66371
# Fold 7 RMSE: 0.66154
# Fold 8 RMSE: 0.66206
# Fold 9 RMSE: 0.66308
# repeated CV RMSE: 0.66271 (std=0.00096)

# Preparing the submission
submission = pd.DataFrame({'p0q0':predictions})
submission.to_csv("xgboost_optim.csv")