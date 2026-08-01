import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from predict import predict_customer

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Banner
st.image(
    "images/banner.png",
    use_container_width=True
)

st.markdown("""
<style>

/* Main background */
.stApp{
    background-color:#F8FAFC;
}

/* Header */
h1{
    color:#0F172A;
    text-align:center;
}

/* Card */
div[data-testid="stMetric"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0 4px 10px rgba(0,0,0,.1);
}

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:10px;
    height:55px;
    font-size:20px;
    font-weight:bold;
}

/* ================= Sidebar ================= */

[data-testid="stSidebar"]{
    background-color:#0F172A !important;
}

/* Sidebar title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
    color:#FFFFFF !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label{
    color:#FFFFFF !important;
    font-weight:600;
}

/* Sidebar markdown */
[data-testid="stSidebar"] p{
    color:#FFFFFF !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] .stRadio label{
    color:#FFFFFF !important;
}

/* Sidebar icons */
[data-testid="stSidebar"] svg{
    fill:white !important;
}


</style>
""", unsafe_allow_html=True)


# Title
st.markdown("""
# 📊 Customer Churn Prediction Dashboard

### Predict customer churn using Machine Learning
""")

st.write(
    "Predict whether a telecom customer is likely to churn using a Machine Learning model."
)

# Sidebar
st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=80
)

st.sidebar.markdown("""
# 📊 Customer Churn

### AI Dashboard
""")
st.sidebar.markdown("---")
st.sidebar.subheader("📑 Navigation")
page = st.sidebar.radio(
    "",
    [
        "🏠 Prediction",
        "📈 Model Evaluation",
        "ℹ️ About"
    ]
)

# Predictiion page
if page == "🏠 Prediction":

    st.header("Customer Details")

    # ==============================
    # KPI Cards
    # ==============================
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric("🤖 Model", "Random Forest")

    with kpi2:
        st.metric("🎯 Accuracy", "77.86%")

    with kpi3:
        st.metric("📈 ROC-AUC", "0.84")

    st.divider()

    # ==============================
    # Customer Input Form
    # ==============================

    col1, col2 = st.columns(2)

    # Left column
    with col1:

        gender = st.selectbox("Gender", ["Male", "Female"])

        senior = st.selectbox(
            "Senior Citizen",
            [0,1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes","No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes","No"]
        )

        tenure = st.slider(
            "Tenure",
            0,
            72,
            24
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes","No"]
        )

        multiple = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    # Right column
    with col2:

        device = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=1500.0
        )

    # Predict button
    st.divider()

    if st.button("🔮 Predict Churn", use_container_width=True):

        # Collect user input
        user_input = {

            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total

        }

        # Make prediction
        result, probability = predict_customer(user_input)

        # Display result
        st.subheader("Prediction Result")

        if "Churn" in result:
            st.markdown("""## 🔴 High Churn Risk""")
        else:
            st.markdown("""## 🟢 Low Churn Risk""")

        st.write(f"**Prediction:** {result}")
        st.metric(
            "Predicted Churn Probability",
            f"{probability*100:.2f}%"
        )
        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        title={"text": "Churn Probability"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"darkblue"},
            "steps":[
                {"range":[0,50],"color":"lightgreen"},
                {"range":[50,75],"color":"gold"},
                {"range":[75,100],"color":"tomato"}
            ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # Risk Indicator
        if probability >= 0.75:
            st.error("⚠️ High Risk Customer")
        elif probability >= 0.50:
            st.warning("🟠 Medium Risk Customer")
        else:
            st.success("🟢 Low Risk Customer")

        # =====================================
        # Business Recommendation
        # =====================================

        st.divider()

        st.subheader("💡 Business Recommendation")

        if probability >= 0.75:

            st.warning("""
        ### Immediate Action Required

        - 📞 Contact the customer immediately.
        - 💰 Offer a loyalty discount.
        - 🎁 Provide exclusive promotional offers.
        - 🤝 Assign a dedicated customer support representative.
        - 📅 Encourage switching to a long-term contract.
        """)

        elif probability >= 0.50:

            st.info("""
        ### Moderate Retention Strategy

        - 🎯 Send personalized promotional offers.
        - 📦 Recommend bundled services.
        - 📧 Follow up with customer satisfaction surveys.
        - 🔍 Monitor customer engagement regularly.
        """)

        else:

            st.success("""
        ### Customer Retention Status

        - ✅ Customer has a low risk of churning.
        - 😊 Continue regular engagement.
        - 🎉 Reward loyalty with occasional offers.
        - 📈 Monitor customer activity as part of routine service.
        """)

        # Download prediction button
        prediction_df = pd.DataFrame({

        "Prediction":[result],

        "Probability (%)":[round(probability*100,2)]

        })

        st.download_button(

            "📥 Download Prediction",

            prediction_df.to_csv(index=False),

            file_name="prediction.csv",

            mime="text/csv"

        )        
            
elif page == "📈 Model Evaluation":

    st.header("📈 Model Performance")

    st.write("Visualizations generated during model training.")

    st.subheader("Dataset Overview")

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric("Customers","7043")

    with c2:
        st.metric("Features","19")

    with c3:
        st.metric("Training Samples","5634")

    with c4:
        st.metric("Test Samples","1409")

   
     
    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Confusion Matrix")
        st.image(
            "images/confusion_matrix.png",
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("ROC Curve")
        st.image(
            "images/roc_curve.png",
            use_container_width=True
        )

    with col2:

        st.subheader("Feature Importance")
        st.image(
            "images/feature_importance.png",
            use_container_width=True
        )

        with st.expander("📘 Model Details"):

            st.write("""
            - Random Forest Classifier
            - Hyperparameter tuning using RandomizedSearchCV
            - SMOTE for balancing classes
            - Cross Validation
            - Feature Importance Analysis
            """)

        st.markdown("---")

        st.info("""
        ### Model Summary

        - **Algorithm:** Random Forest
        - **Accuracy:** 77.86%
        - **ROC-AUC:** 0.84
        - **Class Balancing:** SMOTE
        - **Hyperparameter Tuning:** RandomizedSearchCV
        """)
        
elif page == "ℹ️ About":

    st.header("About This Project")

    c1,c2 = st.columns(2)

    with c1:

        st.success("Random Forest")

        st.success("SMOTE")

        st.success("Streamlit")

        st.success("Scikit-learn")

    with c2:

        st.info("Accuracy : 77.86%")

        st.info("ROC-AUC : 0.84")

        st.info("Cross Validation")

        st.info("Hyperparameter Tuning")

     

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.caption(
    "Developed by Rishabh | Machine Learning | Streamlit | Scikit-learn | 2026"
)             