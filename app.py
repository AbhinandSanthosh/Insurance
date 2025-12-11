import os
import io
import streamlit as st
import pandas as pd
import joblib
from datetime import date, time, datetime

st.set_page_config(page_title="Crime Case Classifier", page_icon="🚨", layout="centered")

RAW_COLS = [
    "Date Reported", "Date of Occurrence", "Time of Occurrence",
    "City", "Crime Code", "Crime Description",
    "Victim Age", "Victim Gender", "Weapon Used",
    "Crime Domain", "Police Deployed"
]

@st.cache_resource
def load_model():
    candidates = ["crime_model.pkl", "svm_model.pkl", "model.pkl"]
    last_err = None
    for path in candidates:
        if os.path.exists(path):
            try:
                m = joblib.load(path)
                return m, path
            except Exception as e:
                last_err = e
    if last_err:
        raise RuntimeError(f"Model load failed: {last_err}")
    raise FileNotFoundError("No model file found. Place crime_model.pkl or svm_model.pkl next to app.py")

try:
    model, model_path = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("🚨 Crime Case Closure Classifier")
st.caption(f"Loaded model: {model_path}  •  Expects raw columns: {', '.join(RAW_COLS)}")

def build_single_row(
    d_reported: date, d_occ: date, t_occ: time,
    city: str, crime_code: str, crime_desc: str,
    victim_age: int, victim_gender: str, weapon_used: str,
    crime_domain: str, police_deployed: int
) -> pd.DataFrame:
    """Build a single-row DataFrame with the SAME raw column names as training."""
    return pd.DataFrame([{
        "Date Reported": datetime.combine(d_reported, time(9, 0)),    # assume 9AM report if not provided
        "Date of Occurrence": datetime.combine(d_occ, t_occ),
        "Time of Occurrence": t_occ.strftime("%H:%M"),
        "City": city,
        "Crime Code": crime_code,
        "Crime Description": crime_desc or "",
        "Victim Age": victim_age,
        "Victim Gender": victim_gender,
        "Weapon Used": weapon_used,
        "Crime Domain": crime_domain,
        "Police Deployed": police_deployed
    }], columns=RAW_COLS)

def predict_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run prediction on a DataFrame with RAW_COLS; returns a copy with predictions."""
    out = df_raw.copy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_raw)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out["pred_class"] = pred
        out["pred_proba_class1"] = proba
    else:
        # probability disabled in SVC; fall back to predict only
        pred = model.predict(df_raw)
        out["pred_class"] = pred
    return out

with st.form("single_input"):
    st.subheader("Single Prediction")
    c1, c2 = st.columns(2)

    with c1:
        d_reported = st.date_input("Date Reported", value=date(2020, 1, 1))
        d_occ = st.date_input("Date of Occurrence", value=date(2020, 1, 1))
        t_occ = st.time_input("Time of Occurrence", value=time(12, 0))
        victim_age = st.number_input("Victim Age", min_value=0, max_value=120, value=30)
        police_deployed = st.number_input("Police Deployed", min_value=0, value=10)

    with c2:
        city = st.text_input("City", "Mumbai")
        crime_code = st.text_input("Crime Code", "IPC-123")
        victim_gender = st.selectbox("Victim Gender", ["Male", "Female", "Other"])
        weapon_used = st.text_input("Weapon Used", "None")
        crime_domain = st.text_input("Crime Domain", "Urban")
        crime_desc = st.text_area("Crime Description", "")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        row = build_single_row(
            d_reported, d_occ, t_occ,
            city, crime_code, crime_desc,
            int(victim_age), victim_gender, weapon_used,
            crime_domain, int(police_deployed)
        )
        preds = predict_df(row)

        st.success(f"Prediction: {'Closed' if int(preds.loc[0,'pred_class'])==1 else 'Not Closed'}")
        if "pred_proba_class1" in preds.columns:
            st.write(f"Probability of Closed: {float(preds.loc[0, 'pred_proba_class1']):.2%}")

        with st.expander("Show model input row"):
            st.dataframe(row)
        with st.expander("Show full prediction output"):
            st.dataframe(preds)
    except Exception as e:
        st.error("Prediction failed. This usually means the saved model is not a full preprocessing pipeline or the input columns differ from training.")
        st.exception(e)

st.divider()
st.subheader("Batch Predictions (CSV or Excel)")

uploaded = st.file_uploader("Upload a file with the raw columns", type=["csv", "xlsx", "xls"])
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_up = pd.read_csv(uploaded)
        else:
            df_up = pd.read_excel(uploaded)

        missing_cols = [c for c in RAW_COLS if c not in df_up.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            preds = predict_df(df_up[RAW_COLS])
            st.success(f"Predicted {len(preds)} rows")
            st.dataframe(preds.head())

            # Download results
            buf = io.BytesIO()
            preds.to_csv(buf, index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error("Batch prediction failed.")
        st.exception(e)

st.caption("Note: The model file should be a scikit-learn Pipeline that performs all preprocessing internally.")