from time import sleep
import streamlit as st
import pandas as pd
import joblib
import re
import os
from datetime import datetime

# Paths
MODEL_PATH = "models/best_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"
LOG_FILE = "logs/input_errors.log"

# Load model + features (fail gracefully if missing)
try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    # st.write("Loaded feature names count:", len(feature_names))
    # st.write("First few features:", feature_names[:10])

except Exception as e:
    # Delay hard failure until prediction time; keep CLI usable for now.
    st.error(f"Error loading feature_names.")
    model = None
    feature_names = None

def log_event(kind, field, value, message):
    """Log invalid input or auto-corrections to a file."""
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {kind.upper()} | Field: {field} | Value: '{value}' | {message}\n")
    except Exception as e:
        # Never crash on logging, but show debug info in console.
        st.error(f"Logging event failed.")

def preprocess_input(sample_dict, feature_names):
    """Convert user inputs into the same format as training data."""
    if feature_names is None:
        print("⚠️ Feature files not found. Please run the training notebook to generate models first.")
        return None
    
    # Convert dict -> DataFrame
    sample = pd.DataFrame([sample_dict])

    # Normalize field names to training-time keys
    if "Cpu" in sample.columns:
        sample = sample.rename(columns={"Cpu": "Cpu_brand"})
    if "Gpu" in sample.columns:
        sample = sample.rename(columns={"Gpu": "Gpu_brand"})
    if "OpSys" in sample.columns:
        sample = sample.rename(columns={"OpSys": "OpSys"})

    # One-hot encode & align
    sample_encoded = pd.get_dummies(sample)
    sample_encoded = sample_encoded.reindex(columns=feature_names, fill_value=0)

    # numeric_columns = sample_encoded.select_dtypes(include=["float64", "int64"]).columns
    # sample_encoded[numeric_columns] = scaler.transform(sample_encoded[numeric_columns])
    return sample_encoded

def predict_price(model, sample_encoded):
    """Predict laptop price given specs; logs issues if any."""
    # Safety Checks
    if model is None:
        st.warning("⚠️ Model not loaded. Please train first.")
        return None
    
    try:
        pred = model.predict(sample_encoded)[0]
        pred = round(float(pred), 4)
        # Log outliers (dataset: ~€100–€6990)
        if pred < 100 or pred > 6990:
            log_event("warn", "PREDICT", str(pred), f"Unrealistic prediction generated: €{pred}.")
        return pred
    
    except Exception as e:
        st.warning("⚠️ Prediction failed. Please check preprocessing.")
        log_event("error", "PREDICT", str(sample_encoded), f"Prediction failure: {e}")
        return None

def final_price(pred, company, typename, touch):
    """
    Adjust predicted laptop price based on range and touchscreen feature.
"""
    # Base adjustment by price range
    if 100 < pred < 400:     # Low-end.
        final_pred = pred * 1.05   # +5%        
        st.info(f"Approximate Price for the {str(company)} {str(typename)}: €{pred:.2f}")
        st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")
        # pred += 50
        # final_pred += 50
        # st.info(f"Approximate Price for the {str(company)} {str(typename)}: €{pred:.2f}")
        # st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")

    elif 400 < pred < 800:  # Mid-range.
        final_pred = pred * 0.98   # -2%
        # Touchscreen price bump (+100)
        # if touch == "No":
        #     st.info(f"Approximate Price for the {str(company)} {str(typename)}: €{pred:.2f}")
        #     st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")
        # else:
        #     pred += 100
        #     final_pred += 100
        st.info(f"Approximate Price for the {str(company)} {str(typename)}: €{pred:.2f}")
        st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")

    elif pred >= 800:     # High-end. So we keep it as is.
        # Touchscreen price bump (+150)
        # if touch == "No":
        #     st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{final_pred:.2f}")
        # else:
        #     final_pred += 150
        st.success(f"Estimated Price for the {str(company)} {str(typename)}: €{pred:.2f}")

    else:
        st.warning("⚠️ Prediction failed. Please check preprocessing.")
        log_event("error", "PREDICT", str(pred), f"Prediction failure")
        return None


# Load the trained model
model = joblib.load("models/best_model.pkl")
# scaler = joblib.load("models/scaler.pkl")

# Streamlit UI
st.title("Laptop Price Prediction App")
st.write("Enter your laptop specifications to predict the price in Euros (€).")

# Choice options
CPU_MAP = {
    # Intel
    "Intel Core i3": "Intel",
    "Intel Core i5": "Intel",
    "Intel Core i7": "Intel",
    "Intel Core i9": "Intel",
    "Intel Pentium": "Intel",
    "Intel Celeron": "Intel",

    # AMD
    "AMD Ryzen 3": "AMD",
    "AMD Ryzen 5": "AMD",
    "AMD Ryzen 7": "AMD",
    "AMD Ryzen 9": "AMD",
    "AMD Athlon": "AMD",

    # Apple & Fallback (Other)
    "Apple M1": "Other",
    "Apple M2": "Other",
    "Other": "Other"
}
GPU_MAP = {
    # Intel
    "Intel HD Graphics": "Intel",
    "Intel UHD Graphics": "Intel",
    "Intel Iris Xe": "Intel",

    # Nvidia
    "Nvidia GeForce GTX 1050": "Nvidia",
    "Nvidia GeForce GTX 1650": "Nvidia",
    "Nvidia GeForce RTX 2060": "Nvidia",
    "Nvidia GeForce RTX 3060": "Nvidia",
    "Nvidia GeForce RTX 4090": "Nvidia",

    # AMD
    "AMD Radeon Vega 8": "AMD",
    "AMD Radeon RX 5600M": "AMD",
    "AMD Radeon RX 6800M": "AMD",

    # Fallback
    "Other": "Other"
}
companies = [
    "Dell",
    "HP", 
    "Lenovo", 
    "Apple", 
    "Asus", 
    "Acer", 
    "MSI",
    "Samsung",
    "Toshiba",
    "Huawei",
    "Microsoft",
    "Chuwi",
    "Xiaomi",
    "Vero",
    "Razer",
    "Mediacom",
    "Google",
    "Fujitsu",
    "LG"
    ]
typenames = [
    "Ultrabook", 
    "Notebook", 
    "Gaming", 
    "2 in 1 Convertible", 
    "Workstation",
    "Netbook"
    ]
osys = [
    "Windows 11",
    "Windows 10",
    "Windows 10 S", 
    "Windows 7", 
    "MacOS",
    "MacOS X",
    "Linux",
    "Android",
    "Chrome OS", 
    "No OS"
    ]


# User inputs
company = st.selectbox("Company", sorted(companies))
typename = st.selectbox("Type", sorted(typenames))
cpu_choice = st.selectbox("CPU Brand", sorted(list(CPU_MAP.keys())))
cpu = CPU_MAP.get(cpu_choice, "Other")  # Default to "Other" if not found.
gpu_choice = st.selectbox("GPU Brand", sorted(list(GPU_MAP.keys())))
gpu = GPU_MAP.get(gpu_choice, "Other")
opsys = st.selectbox("Operating System", sorted(osys))
touch = st.selectbox("Touchscreen", ["Yes", "No"])

ram = st.slider("RAM (GB)", min_value=4, max_value=128, step=4, value=8)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
ssd = st.number_input("SSD size (GB)", min_value=0, max_value=4000, value=512, step=128)
hdd = st.number_input("HDD size (GB)", min_value=0, max_value=6000, value=0, step=500)
flash = st.number_input("Flash Storage (GB)", min_value=0, max_value=1000, value=0, step=128)
hybrid = st.number_input("Hybrid storage (GB)", min_value=0, max_value=2000, value=0, step=500)
inches = st.slider("Screen Size (inches)", min_value=10.0, max_value=18.0, step=0.1, value=12.0)

# Normalizing Company Casing
if company == "HP": company = "Hp"
if company == "MSI": company = "Msi"
if company == "LG": company = "Lg"
if company == "MacOS": company = "macOS"

# Normalize OS
if opsys == "Windows 11": opsys = "Windows 10"

# Predict Button
if st.button("Predict Price"):
    try:
        # Build input DataFrame
        received_data = {
            "Company": company,
            "TypeName": typename,
            "Inches": inches,
            "Ram": ram,
            "Weight": weight,
            "OpSys": opsys,
            "SSD": ssd,
            "HDD": hdd,
            "Hybrid": hybrid,
            "Flash_Storage": flash,
            "Cpu_brand": cpu,
            "Gpu_brand": gpu
        }

        # NOTE: Applying the same preprocessing (target encoding + scaling) here.
        input_data = preprocess_input(received_data, feature_names)

        # st.write("Encoded sample shape:", input_data.shape)
        # st.write("Model expects:", len(feature_names), "features")
        # missing = set(feature_names) - set(input_data.columns)
        # extra = set(input_data.columns) - set(feature_names)
        # st.write("Missing features:", missing)
        # st.write("Unexpected extra features:", extra)

        # Prediction
        with st.spinner("Calculating..."):
            sleep(1.0)
            prediction = predict_price(model, input_data)
            # prediction = model.predict(input_data)[0]
        
        # Sanity bounds (based on your dataset: €100 - €6999)
        if prediction is not None:
            if prediction < 100 or prediction > 6999:  # Bounds based on dataset and logic.
                log_event("warn", "Prediction", str(received_data), f"Unrealistic prediction: {prediction:.2f}")
                st.warning(f"Prediction seems unrealistic (€{prediction:,.2f}). Please re-check your inputs.")
            else:
                st.toast("Prediction ready!")
                sleep(1.0)
                final_price(prediction, company, typename, touch)     
        else:
            st.error(f"Prediction failed. Please try again.")

    except Exception as e:
        log_event("error", "StreamlitApp", str(received_data), f"Prediction failed: {e}")
        st.error(f"Something went wrong while generating the prediction. Please check your inputs and try again.{e}")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        font_size: 12px;
        color: gray;
    }
    </style>
    <div class="footer">© 2025 Group P</div>
    """,
    unsafe_allow_html=True
)