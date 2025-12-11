import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("insurance_model.pkl")

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
    prediction = model.predict(input_data)[0]
    st.subheader("Predicted Insurance Charge")
    st.success(f"${prediction:,.2f}")

    # DEBUG BLOCK - remove after verifying
import os, streamlit as st
st.set_page_config(page_title="DEBUG - Insurance App")
st.markdown("## DEBUG: This is the Insurance app")
st.write("app __file__:", os.path.abspath(__file__))
st.write("cwd:", os.getcwd())
st.write("files in cwd:", sorted(os.listdir(os.getcwd())))
st.stop()   # comment out after confirming
