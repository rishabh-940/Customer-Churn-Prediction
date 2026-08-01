from utils.preprocessing import *
from utils.model_utils import *
from utils.evaluation import *

import joblib


def main():

    print("Training Started...")

    # Load data
    df = load_data("dataset/Telco_Customer_Churn.csv")

    # Clean data
    df = clean_data(df)

    # Encode data
    df, encoders = encode_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(
        X_train,
        y_train
    )

    # Compare models
    models = get_models()

    results = compare_models(
        models,
        X_train_smote,
        y_train_smote
    )

    # Select best model
    best_model_name = select_best_model(results)

    # Train model
    if best_model_name == "Random Forest":
        print("\nTuning Random Forest...")
        model = tune_random_forest(
            X_train_smote,
            y_train_smote
        )
    else:
        model = train_model(
            best_model_name,
            models,
            X_train_smote,
            y_train_smote
        )
    # -----------------------------
    # Evaluation
    # -----------------------------
    create_output_directory()

    y_pred, y_prob, metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    plot_roc_curve(
        y_test,
        y_prob
    )

    plot_feature_importance(
        model,
        X_train.columns
    )

    # Save model
    save_model(model, "model/churn_model.pkl")
    save_encoders(encoders, "model/label_encoders.pkl")

    joblib.dump(
        X_train.columns.tolist(),
        "model/feature_columns.pkl"
    )

    print("Training Completed Successfully!")


if __name__ == "__main__":
    main()