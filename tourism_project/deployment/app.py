# app.py
# Streamlit prediction app for the Wellness Tourism Package.
# Loads the best XGBoost model from the Hugging Face Model Hub
# and returns a purchase probability for each customer profile entered.

import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────
HF_USER        = "kavumadouglas"
MODEL_REPO     = "tourism-prediction-model"
model_repo_id  = f"{HF_USER}/{MODEL_REPO}"

# ── Load model (cached so it only downloads once per session) ─
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=model_repo_id,
        filename="best_model.joblib",
        repo_type="model"
    )
    return joblib.load(model_path)

model = load_model()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Tourism Package Predictor", page_icon="🌍", layout="wide")
st.title("🌍 Wellness Tourism Package — Purchase Predictor")
st.write(
    "Enter a customer profile below to predict whether they are likely "
    "to purchase the new **Wellness Tourism Package**."
)
st.divider()

# ── Input form ────────────────────────────────────────────────
st.subheader("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age            = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier      = st.selectbox("City Tier", [1, 2, 3])
    gender         = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation     = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    designation    = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)

with col2:
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    num_persons     = st.number_input("Number Of Persons Visiting", min_value=1, max_value=10, value=2)
    num_trips       = st.number_input("Avg Number Of Trips Per Year", min_value=0, max_value=20, value=2)
    preferred_star  = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
    passport        = st.selectbox("Holds Passport?", ["No", "Yes"])
    own_car         = st.selectbox("Owns a Car?", ["No", "Yes"])
    num_children    = st.number_input("Number Of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)

st.subheader("Customer Interaction Data")
col3, col4 = st.columns(2)

with col3:
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_score     = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

with col4:
    num_followups  = st.number_input("Number Of Followups", min_value=0, max_value=10, value=3)
    duration_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=60, value=15)

st.divider()

# ── Prediction ────────────────────────────────────────────────
if st.button("🔍 Predict Purchase Likelihood", use_container_width=True):
    input_df = pd.DataFrame([{
        "Age":                      age,
        "TypeofContact":            type_of_contact,
        "CityTier":                 city_tier,
        "DurationOfPitch":          duration_pitch,
        "Occupation":               occupation,
        "Gender":                   gender,
        "NumberOfPersonVisiting":   num_persons,
        "NumberOfFollowups":        num_followups,
        "ProductPitched":           product_pitched,
        "PreferredPropertyStar":    preferred_star,
        "MaritalStatus":            marital_status,
        "NumberOfTrips":            num_trips,
        "Passport":                 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore":   pitch_score,
        "OwnCar":                   1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation":              designation,
        "MonthlyIncome":            monthly_income,
    }])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(
            f"✅ **Likely to purchase** the Wellness Tourism Package\n\n"
            f"Purchase probability: **{probability:.1%}**"
        )
    else:
        st.warning(
            f"❌ **Unlikely to purchase** the Wellness Tourism Package\n\n"
            f"Purchase probability: **{probability:.1%}**"
        )

    st.caption("Prediction powered by XGBoost · Model hosted on Hugging Face Model Hub")
