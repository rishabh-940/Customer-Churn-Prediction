import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


# Load dataset
def load_data(file_path):
    """
    Load the customer churn dataset.

    Parameters:
        file_path (str): Path to CSV file

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(file_path)

# Clean dataset
def clean_data(df):
    """
    Clean dataset before training.
    """

    # Remove customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Replace blank values
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")

    # Convert to float
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    return df

# Encode numerical culumns
def encode_features(df):
    """
    Encode categorical columns using LabelEncoder.

    Returns:
        encoded dataframe,
        dictionary of encoders
    """

    encoders = {}

    object_columns = df.select_dtypes(include="object").columns

    for column in object_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    return df, encoders

# Split dataset
def split_data(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

# Apply SMOTE
def apply_smote(X_train, y_train):

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    return X_train_smote, y_train_smote

