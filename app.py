# app.py - Insurance Charges Prediction (robust loader + debug)
import os
import traceback

import streamlit as st
import pandas as pd

# Page config (call before other Streamlit calls that affect layout)
st.set_page_config(page_title="Insurance Charges Prediction", layout="wide")

# --- Optional: small debug toggle (uncheck in production) ---
show_debug = st.sidebar.checkbox("Show debug info", value=False)

if show_debug:
    st.markdown("### DEBUG: runtime info")
    st.write("app __file__:", os.path.abspath(__file__))
    st.write("cwd:", os.getcwd())
    st.write("files in cwd:", sorted(os.listdir(os.getcwd())))

# --- Model loading (try cloudpickle, fall back to joblib with clear errors) ---
MODEL_FILENAME = "insurance_model.pkl"
model = None
model_path = os.path.join(os.getcwd(), MODEL_FILENAME)

if not os.path.exists(model_path):
    st.error(f"Model file not found: `{MODEL_FILENAME}`. Please upload it to the app folder.")
    if show_debug:
        st.stop()
    else:
        st.stop()

# Try to load with cloudpickle (more robust across environments)
try:
    import cloudpickle

    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
    if show_debug:
        st.success("Model loaded with cloudpickle.")
except Exception as e_cloud:
    # If cloudpickle fails, try joblib as a fallback, but show traceback
    st.warning("cloudpickle load failed, trying joblib.load as a fallback...")
    if show_debug:
        st.text("cloudpickle error:")
        st.text(traceback.format_exc())

    try:
        import joblib

        model = joblib.load(model_path)
        if show_debug:
            st.success("Model loaded with joblib.")
    except Exception as e_joblib:
        st.error("Failed to load model using both cloudpickle and joblib. See details below.")
        st.text("cloudpickle error:")
        st.text("".join(traceback.format_exception(None, e_cloud, e_cloud.__traceback__)))
        st.text("joblib error:")
        st.text("".join(traceback.format_exception(None, e_joblib, e_joblib.__traceback__)))
        st.stop()

# --- App UI ---
st.title("Insurance Charges Prediction")
st.write("This app predicts medical insurance charges based on user details.")

# -------- Inputs (match your features exactly) --------
age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

children = st.number_input(
    "Number of Children", min_value=0, max_value=10, value=0, step=1
)

# These values MUST match those in your CSV (check df['sex'].unique())
sex = st.selectbox("Sex", ["male", "female"])

# These values MUST match df['smoker'].unique()
smoker = st.selectbox("Smoker", ["yes", "no"])

# These values MUST match df['region'].unique()
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Create input DataFrame with SAME columns used in training
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

st.subheader("Your Input")
st.write(input_data)

# -------- Prediction --------
if st.button("Predict Insurance Charge"):
    try:
        prediction = model.predict(input_data)[0]
        st.subheader("Predicted Insurance Charge")
        st.success(f"${prediction:,.2f}")
    except Exception as e:
        st.error("Prediction failed. See error details below.")
        st.text("".join(traceback.format_exception(None, e, e.__traceback__)))
        if show_debug:
            st.stop()
