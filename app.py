import joblib
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Mental Health Trend Predictor", page_icon="🧠", layout="centered"
)


# Load model with caching
@st.cache_resource
def load_pipeline():
    return joblib.load("mental_health_rf_pipeline.pkl")


try:
    model = load_pipeline()
    st.title("🧠 Mental Health Trend Predictor")
    st.write(
        "Enter population disorder prevalence rates to predict estimated timeframe indicators."
    )

    # Input Form
    with st.form("prediction_form"):
        code = st.text_input("Country / Entity Code", value="USA")
        schizophrenia = st.number_input(
            "Schizophrenia Share (%)", min_value=0.0, max_value=100.0, value=0.25
        )
        depressive = st.number_input(
            "Depressive Disorders Share (%)",
            min_value=0.0,
            max_value=100.0,
            value=3.80,
        )
        anxiety = st.number_input(
            "Anxiety Disorders Share (%)",
            min_value=0.0,
            max_value=100.0,
            value=4.50,
        )
        bipolar = st.number_input(
            "Bipolar Disorders Share (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.85,
        )

        submit_btn = st.form_submit_button("Predict Estimated Year")

    if submit_btn:
        # Format inputs into dataframe structure expected by column transformers
        input_df = pd.DataFrame(
            [
                {
                    "Code": code,
                    "Schizophrenia disorders (share of population)": schizophrenia,
                    "Depressive disorders (share of population)": depressive,
                    "Anxiety disorders (share of population)": anxiety,
                    "Bipolar disorders (share of population)": bipolar,
                }
            ]
        )

        # Generate prediction
        predicted_year = model.predict(input_df)[0]

        st.success(f"**Predicted Estimated Year:** {predicted_year:.2f}")

except Exception as e:
    st.error(f"Error loading model pipeline: {str(e)}")