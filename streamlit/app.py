import streamlit as st
import pandas as pd
import joblib
import os

#  Using Streamlit to Load trained model

@st.cache_resource
def load_model():
    model_path = "../models/laptop_price_model.pkl"   
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found. Please train and save the model first.")
        return None
    return joblib.load(model_path)

model = load_model()




st.set_page_config(page_title="💻 Laptop Price Predictor", layout="centered")
st.title("💻 Laptop Price Predictor")
st.markdown("Enter laptop details below to predict its price in Euros (€).")

# Input fields
company = st.selectbox("Company", ["Apple", "HP", "Dell", "Lenovo", "Asus", "Acer", "MSI", "Other"])
typename = st.selectbox("Type", ["Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible", "Workstation"])
inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, step=0.1)
cpu = st.selectbox("CPU", ["Intel Core i5 2.3GHz", "Intel Core i5 1.8GHz", "Intel Core i7 2.7GHz", "Intel Core i7 3.1GHz", "Other"])
ram = st.number_input("RAM (GB)", min_value=2, max_value=64, step=2)
gpu = st.selectbox("GPU", ["Intel HD Graphics 620", "Intel Iris Plus Graphics 640", "AMD Radeon Pro 455", "Nvidia GTX 1050", "Other"])
opsys = st.selectbox("Operating System", ["Windows", "macOS", "Linux", "No OS", "Other"])
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)

ssd = st.number_input("SSD (GB)", min_value=0, max_value=2000, step=128)
hdd = st.number_input("HDD (GB)", min_value=0, max_value=2000, step=128)
hybrid = st.number_input("Hybrid (GB)", min_value=0, max_value=2000, step=128)
flash_storage = st.number_input("Flash Storage (GB)", min_value=0, max_value=512, step=32)
touchscreen = st.selectbox("Touchscreen", [0, 1])
x_res = st.number_input("X Resolution", min_value=800, max_value=4000, step=100)
y_res = st.number_input("Y Resolution", min_value=600, max_value=2500, step=100)


# Predict Button

if st.button("🔮 Predict Price"):
    if model is not None:
        # Prepare input DataFrame
        input_data = pd.DataFrame([{
            "Company": company,
            "TypeName": typename,
            "Inches": inches,
            "Cpu": cpu,
            "Ram": ram,
            "Gpu": gpu,
            "OpSys": opsys,
            "Weight": weight,
            "SSD": ssd,
            "HDD": hdd,
            "Hybrid": hybrid,
            "Flash_Storage": flash_storage,
            "Touchscreen": touchscreen,
            "X_resolution": x_res,
            "Y_resolution": y_res
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: **€{prediction:,.2f}**")
    else:
        st.error(" No model available. Please train and save a model first.")
