import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_df(df):

    df = df[["train", "gare", "date", "arret", "p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]]
    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    

    # Extract useful date features for later analysis
    # df['year'] = df['date'].dt.year
    # df['month'] = df['date'].dt.month
    # df['day'] = df['date'].dt.day
    # df['dayofweek'] = df['date'].dt.dayofweek
    # df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    # Combine 'gare' (station name) and 'arret' (stop number) into one composite feature.
    # This captures the joint information that may be important for the prediction.
    df['gare_arret'] = df['gare'].astype(str) + "_" + df['arret'].astype(str)

    # One-hot encode the composite variable.
    # (Depending on your dataset, consider if one-hot encoding is the best approach.)
    df = pd.get_dummies(df, columns=['gare_arret'], prefix='ga', drop_first=True)
    # df = pd.get_dummies(df, columns=['train'], drop_first=True)

    # Optionally, drop the original columns that are now encoded and the raw date column
    df = df.drop(columns=['train', 'gare', 'arret', 'date'])

    return df





if __name__ == "__main__":
    main()
