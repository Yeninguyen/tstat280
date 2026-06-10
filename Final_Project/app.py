import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── App title (must be first Streamlit command) ──
st.set_page_config(page_title="Home Price Estimator", layout="centered")
st.title("Home Price Estimator")

# ── Block 1: Load model, scaler, and column names ──
# Save LASSO model (not the categorical model)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
training_columns = joblib.load("columns.pkl")

# ── Block 2: Get input from user ──
with st.sidebar:
    st.header("Property Details")

    beds       = st.slider("Bedrooms", 1, 9, 3)
    bathrooms  = st.slider("Bathrooms", 1.0, 6.0, 2.0, step=0.5)
    house_size = st.slider("House Size (sqft)", 300, 7000, 1600, step=50)
    year_built = st.slider("Year Built", 1900, 2025, 1990)
    lot_size = st.number_input("Lot Size (sqft)", min_value=500, max_value=100000, value=6000, step=500)
    hoa_fee = st.number_input("Monthly HOA Fee ($)", min_value=0, max_value=5000, value=0, step=25)

    mls_area = st.selectbox("MLS Area", [
        "13 - North Tacoma (base)",
        "14 - North Tacoma", "15 - North Tacoma", "16 - North Tacoma",
        "17 - North Tacoma", "18 - North Tacoma", "19 - North Tacoma",
        "20 - North Tacoma", "21 - North Tacoma", "22 - North Tacoma",
        "23 - North Tacoma", "24 - North Tacoma", "25 - North Tacoma",
        "26 - Central Tacoma", "27 - Central Tacoma", "28 - Central Tacoma",
        "29 - Central Tacoma", "30 - Central Tacoma", "31 - West Tacoma",
        "32 - University Place", "33 - University Place", "34 - University Place",
        "35 - University Place", "36 - Lakewood", "37 - Lakewood",
        "38 - Lakewood", "39 - Lakewood", "43 - Steilacoom",
        "45 - South Tacoma", "46 - South Tacoma", "47 - South Tacoma",
        "48 - South Tacoma", "49 - South Tacoma", "50 - South Tacoma",
        "51 - South Tacoma", "52 - South Tacoma", "53 - South Tacoma",
        "54 - South Tacoma", "55 - Southeast Tacoma", "56 - Southeast Tacoma",
        "57 - Southeast Tacoma", "58 - Southeast Tacoma", "59 - Southeast Tacoma",
        "60 - Southeast Tacoma", "61 - Southeast Tacoma", "62 - Southeast Tacoma",
        "63 - Parkland", "64 - Parkland/Midland", "65 - Parkland",
        "66 - Parkland", "67 - Parkland", "68 - Parkland", "69 - Parkland",
        "70 - Fife", "71 - Milton", "79 - Puyallup", "80 - Puyallup",
        "87 - Puyallup", "88 - Puyallup", "89 - Graham/Frederickson",
        "94 - Browns Point", "95 - Browns Point", "99 - Spanaway", "_99",
    ])
    cooling = st.selectbox("Cooling", [
    "Ductless Mini Split",
    "Forced Air",
    "Heat Pump",
    "None",
    "Other"
])
    heating = st.selectbox("Heating", [
    "Ductless Mini Split",
    "Forced Air",
    "Heat Pump",
    "Wall Unit",
    "Other"
])
    sewer = st.selectbox("Sewer", [
    "Septic Tank",
    "Sewer Connected"
])
    parking = st.selectbox("Parking Features", [
    "Detached Garage",
    "Off Street Parking",
    "Other"
])
    latitude = st.number_input("Latitude", value=47.25)
    longitude = st.number_input("Longitude", value=-122.44)

# ── Block 3: Prepare data and predict ──

# Step 1: Build DataFrame
input_data = {
    "beds": beds,
    "Bathrooms": bathrooms,
    "Latitude": latitude,
    "Longitude": longitude,
    "house_size": house_size,
    "Lot Size Square Feet": lot_size,
    "Year Built": year_built,
    "HOA Fee": hoa_fee,
    "MLS Area": mls_area,
    "Cooling": cooling,
    "Heating": heating,
    "Sewer": sewer,
    "Parking Features": parking,
}
input_df = pd.DataFrame([input_data])

# Step 2: Create dummy variables matching training
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

# Step 3: Scale
input_scaled = scaler.transform(input_encoded)

# Step 4: Predict
log_prediction = model.predict(input_scaled)[0]
prediction = np.exp(log_prediction)

# ── Block 4: Display result ──
st.subheader("Estimated Home Value")
st.metric(
    label="Predicted Price",
    value=f"${prediction:,.2f}"
)