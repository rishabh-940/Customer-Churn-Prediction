import joblib
import pandas as pd

# Load saved models
# Load saved model artifacts
def load_artifacts():
   
    model = joblib.load("model/churn_model.pkl")
    encoders = joblib.load("model/label_encoders.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")

    return model, encoders, feature_columns

  
# Load artifacts once
model, encoders, feature_columns = load_artifacts()

# Encode user input
def preprocess_input(input_data):
    """
    Convert user input into model-ready format.
    """

    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for column, encoder in encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column])

    # Ensure feature order matches training
    df = df[feature_columns]

    return df

# Prediction function
def predict_customer(input_data):

    processed_data = preprocess_input(input_data)

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0][1]

    # Convert prediction to readable text
    if prediction == 1:
        result = "Customer is likely to Churn"
    else:
        result = "Customer is likely to Stay"

    return result, probability

# Testing
if __name__ == "__main__":

    sample_customer = {

        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.75,
        "TotalCharges": 1077.00
    }

    result, probability = predict_customer(sample_customer)
    print("\nPrediction Result")
    print("-------------------------")
    print(result)
    print(f"Churn Probability: {probability*100:.2f}%")
